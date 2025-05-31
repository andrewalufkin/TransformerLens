"""Devices.

Utilities to get the correct device, and assist in distributing model layers across multiple
devices.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Union, Dict, List, Tuple
from dataclasses import dataclass

import torch
from torch import nn

import transformer_lens

AvailableDeviceMemory = list[tuple[int, int]]
"""
This type is passed around between different CUDA memory operations.
The first entry of each tuple will be the device index.
The second entry will be how much memory is currently available.
"""


def calculate_available_device_cuda_memory(i: int) -> int:
    """Calculates how much memory is available at this moment for the device at the indicated index

    Args:
        i (int): The index we are looking at

    Returns:
        int: How memory is available
    """
    total = torch.cuda.get_device_properties(i).total_memory
    allocated = torch.cuda.memory_allocated(i)
    return total - allocated


def determine_available_memory_for_available_devices(max_devices: int) -> AvailableDeviceMemory:
    """Gets all available CUDA devices with their current memory calculated

    Returns:
        AvailableDeviceMemory: The list of all available devices with memory precalculated
    """
    devices = []
    for i in range(max_devices):
        devices.append((i, calculate_available_device_cuda_memory(i)))

    return devices


def sort_devices_based_on_available_memory(devices: AvailableDeviceMemory) -> AvailableDeviceMemory:
    """Sorts all available devices with devices with the most available memory returned first

    Args:
        devices (AvailableDeviceMemory): All available devices with memory calculated

    Returns:
        AvailableDeviceMemory: The same list of passed through devices sorted with devices with most
        available memory first
    """
    return sorted(devices, key=lambda x: x[1], reverse=True)


def get_best_available_cuda_device(max_devices: Optional[int] = None) -> torch.device:
    """Gets whichever cuda device has the most available amount of memory for use

    Raises:
        EnvironmentError: If there are no available devices, this will error out

    Returns:
        torch.device: The specific device that should be used
    """
    max_devices = max_devices if max_devices is not None else torch.cuda.device_count()
    devices = determine_available_memory_for_available_devices(max_devices)

    if len(devices) <= 0:
        raise EnvironmentError(
            "TransformerLens has been configured to use CUDA, but no available devices are present"
        )

    sorted_devices = sort_devices_based_on_available_memory(devices=devices)

    return torch.device("cuda", sorted_devices[0][0])


def get_best_available_device(cfg: "transformer_lens.HookedTransformerConfig") -> torch.device:
    """Gets the best available device to be used based on the passed in arguments

    Args:
        device (Union[torch.device, str]): Either the existing torch device or the string identifier

    Returns:
        torch.device: The best available device
    """
    assert cfg.device is not None
    device = torch.device(cfg.device)

    if device.type == "cuda":
        return get_best_available_cuda_device(cfg.n_devices)
    else:
        return device


@dataclass
class ModuleInfo:
    """Information about a transformer module for device allocation."""
    name: str
    parameter_count: int
    is_transformer_block: bool = False
    block_index: Optional[int] = None
    
    @property
    def memory_mb(self) -> float:
        """Calculate memory usage in MB (params + activation overhead)."""
        # 4 bytes per float32 parameter (adjust for different dtypes if needed)
        param_memory_mb = (self.parameter_count * 4) / (1024 * 1024)
        
        # Add activation overhead (2.5x for forward + backward pass estimation)
        activation_overhead = param_memory_mb * 2.5
        
        return param_memory_mb + activation_overhead


class TransformerDeviceAllocator:
    """
    Advanced device allocation for TransformerLens models.
    
    Provides sequential allocation strategy that keeps transformer blocks together
    on the same GPU until memory constraints force a move to the next GPU.
    Also supports greedy round-robin allocation for backward compatibility.
    """
    
    def __init__(self, memory_safety_factor: float = 0.9):
        """
        Initialize the device allocator.
        
        Args:
            memory_safety_factor: Use only this fraction of available GPU memory (0.9 = 90%)
        """
        self.memory_safety_factor = memory_safety_factor
        self.logger = logging.getLogger(__name__)
    
    def get_model_modules_info(self, cfg: "transformer_lens.HookedTransformerConfig") -> List[ModuleInfo]:
        """
        Extract module information from a TransformerLens config for allocation.
        
        Args:
            cfg: HookedTransformerConfig with model parameters
            
        Returns:
            List of ModuleInfo objects representing the model structure
        """
        modules = []
        
        # Calculate parameters per transformer block
        # Each block typically contains: attention + MLP + layer norms
        attention_params = (
            cfg.d_model * cfg.d_head * cfg.n_heads * 4 +  # W_Q, W_K, W_V, W_O
            cfg.d_head * cfg.n_heads * 4  # bias terms if present
        )
        
        mlp_params = (
            cfg.d_model * cfg.d_mlp * 2 +  # W_in, W_out  
            cfg.d_mlp + cfg.d_model  # bias terms
        )
        
        layernorm_params = cfg.d_model * 2  # Two layer norms per block
        
        block_params = attention_params + mlp_params + layernorm_params
        
        # Add transformer blocks
        for i in range(cfg.n_layers):
            modules.append(ModuleInfo(
                name=f"blocks.{i}",
                parameter_count=block_params,
                is_transformer_block=True,
                block_index=i
            ))
        
        # Add other modules (embeddings, final layer norm, unembedding)
        embedding_params = cfg.d_vocab * cfg.d_model
        modules.append(ModuleInfo("embed", embedding_params))
        
        if hasattr(cfg, 'n_ctx') and cfg.n_ctx:
            pos_embed_params = cfg.n_ctx * cfg.d_model
            modules.append(ModuleInfo("pos_embed", pos_embed_params))
        
        modules.append(ModuleInfo("ln_final", cfg.d_model))
        modules.append(ModuleInfo("unembed", cfg.d_vocab * cfg.d_model))
        
        return modules
    
    def allocate_model_devices(
        self,
        cfg: "transformer_lens.HookedTransformerConfig",
        strategy: str = "sequential",
        device_map: Optional[Dict[str, str]] = None,
        max_devices: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Allocate model components across available devices.
        
        Args:
            cfg: HookedTransformerConfig with model parameters
            strategy: "sequential" (default) or "greedy" for allocation strategy
            device_map: Optional user-specified device assignments
            max_devices: Maximum number of devices to use
            
        Returns:
            Dictionary mapping module names to device strings
        """
        device_map = device_map or {}
        
        if cfg.device == "cpu":
            # All modules go to CPU
            modules = self.get_model_modules_info(cfg)
            return {module.name: "cpu" for module in modules}
        
        # Get available GPU memory
        max_devices = max_devices or cfg.n_devices or torch.cuda.device_count()
        gpu_memory_info = self._get_gpu_memory_info(max_devices)
        
        if strategy == "sequential":
            return self._allocate_sequential(cfg, gpu_memory_info, device_map)
        elif strategy == "greedy":
            return self._allocate_greedy(cfg, gpu_memory_info, device_map)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'sequential' or 'greedy'")
    
    def _get_gpu_memory_info(self, max_devices: int) -> Dict[str, float]:
        """Get available memory for each GPU in GB."""
        gpu_memory_gb = {}
        
        for i in range(min(max_devices, torch.cuda.device_count())):
            available_bytes = calculate_available_device_cuda_memory(i)
            available_gb = (available_bytes / (1024**3)) * self.memory_safety_factor
            gpu_memory_gb[f"cuda:{i}"] = available_gb
            
        return gpu_memory_gb
    
    def _allocate_sequential(
        self,
        cfg: "transformer_lens.HookedTransformerConfig",
        gpu_memory_gb: Dict[str, float],
        user_device_map: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Sequential allocation: keep transformer blocks together until memory limit reached.
        """
        modules = self.get_model_modules_info(cfg)
        
        # Convert GB to MB and track usage
        gpu_memory_mb = {gpu: gb * 1024 for gpu, gb in gpu_memory_gb.items()}
        gpu_usage_mb = {gpu: 0.0 for gpu in gpu_memory_mb.keys()}
        available_gpus = sorted(gpu_memory_mb.keys())
        
        allocation_map = {}
        
        # Phase 1: Handle user-pinned modules
        self._handle_user_pinned_modules(
            user_device_map, modules, gpu_usage_mb, gpu_memory_mb, allocation_map
        )
        
        # Phase 2: Allocate transformer blocks sequentially
        current_gpu_idx = 0
        current_gpu = available_gpus[current_gpu_idx] if available_gpus else "cpu"
        
        transformer_blocks = [m for m in modules if m.is_transformer_block]
        transformer_blocks.sort(key=lambda x: x.block_index or 0)
        
        for block in transformer_blocks:
            if block.name in allocation_map:
                continue  # Skip user-pinned blocks
            
            if current_gpu == "cpu" or self._can_fit(current_gpu, block, gpu_usage_mb, gpu_memory_mb):
                allocation_map[block.name] = current_gpu
                if current_gpu != "cpu":
                    gpu_usage_mb[current_gpu] += block.memory_mb
            else:
                # Move to next GPU
                current_gpu_idx += 1
                if current_gpu_idx >= len(available_gpus):
                    self.logger.warning("Model exceeds available GPU memory, some modules will use CPU")
                    current_gpu = "cpu"
                else:
                    current_gpu = available_gpus[current_gpu_idx]
                
                allocation_map[block.name] = current_gpu
                if current_gpu != "cpu":
                    gpu_usage_mb[current_gpu] += block.memory_mb
        
        # Phase 3: Allocate remaining modules
        remaining_modules = [m for m in modules if not m.is_transformer_block]
        for module in remaining_modules:
            if module.name not in allocation_map:
                best_gpu = self._find_best_fit_gpu(module, gpu_usage_mb, gpu_memory_mb)
                allocation_map[module.name] = best_gpu
                if best_gpu != "cpu":
                    gpu_usage_mb[best_gpu] += module.memory_mb
        
        self._log_allocation_summary(allocation_map, gpu_usage_mb, gpu_memory_mb)
        return allocation_map
    
    def _allocate_greedy(
        self,
        cfg: "transformer_lens.HookedTransformerConfig",
        gpu_memory_gb: Dict[str, float],
        user_device_map: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Greedy round-robin allocation for backward compatibility.
        """
        modules = self.get_model_modules_info(cfg)
        
        # Convert GB to MB and track usage
        gpu_memory_mb = {gpu: gb * 1024 for gpu, gb in gpu_memory_gb.items()}
        gpu_usage_mb = {gpu: 0.0 for gpu in gpu_memory_mb.keys()}
        available_gpus = sorted(gpu_memory_mb.keys())
        
        allocation_map = {}
        
        # Phase 1: Handle user-pinned modules
        self._handle_user_pinned_modules(
            user_device_map, modules, gpu_usage_mb, gpu_memory_mb, allocation_map
        )
        
        # Phase 2: Allocate transformer blocks in round-robin fashion
        current_gpu_idx = 0
        transformer_blocks = [m for m in modules if m.is_transformer_block]
        transformer_blocks.sort(key=lambda x: x.block_index or 0)
        
        for block in transformer_blocks:
            if block.name in allocation_map:
                continue  # Skip user-pinned blocks
            
            # Round-robin: try each GPU until one fits
            attempts = 0
            while attempts < len(available_gpus):
                current_gpu = available_gpus[current_gpu_idx]
                
                if self._can_fit(current_gpu, block, gpu_usage_mb, gpu_memory_mb):
                    allocation_map[block.name] = current_gpu
                    gpu_usage_mb[current_gpu] += block.memory_mb
                    break
                
                # Move to next GPU in round-robin
                current_gpu_idx = (current_gpu_idx + 1) % len(available_gpus)
                attempts += 1
            else:
                # No GPU can fit this block
                self.logger.warning(f"Block {block.name} doesn't fit on any GPU, using CPU")
                allocation_map[block.name] = "cpu"
            
            # Move to next GPU for next block (round-robin)
            current_gpu_idx = (current_gpu_idx + 1) % len(available_gpus)
        
        # Phase 3: Allocate remaining modules
        remaining_modules = [m for m in modules if not m.is_transformer_block]
        for module in remaining_modules:
            if module.name not in allocation_map:
                best_gpu = self._find_best_fit_gpu(module, gpu_usage_mb, gpu_memory_mb)
                allocation_map[module.name] = best_gpu
                if best_gpu != "cpu":
                    gpu_usage_mb[best_gpu] += module.memory_mb
        
        self._log_allocation_summary(allocation_map, gpu_usage_mb, gpu_memory_mb)
        return allocation_map
    
    def _handle_user_pinned_modules(
        self,
        user_device_map: Dict[str, str],
        modules: List[ModuleInfo],
        gpu_usage_mb: Dict[str, float],
        gpu_memory_mb: Dict[str, float],
        allocation_map: Dict[str, str]
    ):
        """Handle user-specified device assignments."""
        for module_name, device in user_device_map.items():
            module = next((m for m in modules if m.name == module_name), None)
            if module is None:
                self.logger.warning(f"User-pinned module '{module_name}' not found in model")
                continue
            
            if device != "cpu" and device not in gpu_memory_mb:
                raise ValueError(f"User-specified device '{device}' not available")
            
            if device != "cpu" and not self._can_fit(device, module, gpu_usage_mb, gpu_memory_mb):
                raise ValueError(
                    f"User-pinned module '{module_name}' ({module.memory_mb:.2f} MB) "
                    f"cannot fit on device '{device}'"
                )
            
            allocation_map[module.name] = device
            if device != "cpu":
                gpu_usage_mb[device] += module.memory_mb
    
    def _can_fit(
        self,
        gpu: str,
        module: ModuleInfo,
        gpu_usage_mb: Dict[str, float],
        gpu_memory_mb: Dict[str, float]
    ) -> bool:
        """Check if a module can fit on the specified GPU."""
        if gpu == "cpu":
            return True
        available_mb = gpu_memory_mb[gpu] - gpu_usage_mb[gpu]
        return module.memory_mb <= available_mb
    
    def _find_best_fit_gpu(
        self,
        module: ModuleInfo,
        gpu_usage_mb: Dict[str, float],
        gpu_memory_mb: Dict[str, float]
    ) -> str:
        """Find the GPU with the least available memory that can still fit the module."""
        best_gpu = "cpu"  # Fallback to CPU
        min_remaining_memory = float('inf')
        
        for gpu in gpu_memory_mb.keys():
            if self._can_fit(gpu, module, gpu_usage_mb, gpu_memory_mb):
                remaining = gpu_memory_mb[gpu] - gpu_usage_mb[gpu] - module.memory_mb
                if remaining < min_remaining_memory:
                    min_remaining_memory = remaining
                    best_gpu = gpu
        
        return best_gpu
    
    def _log_allocation_summary(
        self,
        allocation_map: Dict[str, str],
        gpu_usage_mb: Dict[str, float],
        gpu_memory_mb: Dict[str, float]
    ):
        """Log final allocation summary."""
        self.logger.info("=== Device Allocation Summary ===")
        
        # Log GPU usage
        for gpu in sorted(gpu_memory_mb.keys()):
            usage_pct = (gpu_usage_mb[gpu] / gpu_memory_mb[gpu]) * 100
            self.logger.info(
                f"{gpu}: {gpu_usage_mb[gpu]:.1f} MB / {gpu_memory_mb[gpu]:.1f} MB "
                f"({usage_pct:.1f}%)"
            )
        
        # Log module distribution
        gpu_allocations = {}
        for module_name, gpu in allocation_map.items():
            if gpu not in gpu_allocations:
                gpu_allocations[gpu] = []
            gpu_allocations[gpu].append(module_name)
        
        for gpu in sorted(gpu_allocations.keys()):
            modules = gpu_allocations[gpu]
            blocks = [m for m in modules if m.startswith('blocks.')]
            others = [m for m in modules if not m.startswith('blocks.')]
            
            self.logger.info(f"{gpu}: {len(blocks)} blocks + {len(others)} other modules")


# Global allocator instance
_allocator = TransformerDeviceAllocator()


def allocate_model_devices(
    cfg: "transformer_lens.HookedTransformerConfig",
    strategy: str = "sequential",
    device_map: Optional[Dict[str, str]] = None,
    max_devices: Optional[int] = None
) -> Dict[str, str]:
    """
    Allocate model components across available devices.
    
    This is the main public API for device allocation in TransformerLens.
    
    Args:
        cfg: HookedTransformerConfig with model parameters
        strategy: "sequential" (default) or "greedy" for allocation strategy
        device_map: Optional user-specified device assignments  
        max_devices: Maximum number of devices to use
        
    Returns:
        Dictionary mapping module names to device strings
        
    Example:
        >>> cfg = HookedTransformerConfig(n_layers=12, d_model=768, n_devices=2)
        >>> allocation = allocate_model_devices(cfg, strategy="sequential")
        >>> print(allocation)
        {'blocks.0': 'cuda:0', 'blocks.1': 'cuda:0', ..., 'blocks.8': 'cuda:1', ...}
    """
    return _allocator.allocate_model_devices(cfg, strategy, device_map, max_devices)


def get_device_for_block_index(
    index: int,
    cfg: "transformer_lens.HookedTransformerConfig",
    device: Optional[Union[torch.device, str]] = None,
):
    """
    Determine the device for a given layer index based on the model configuration.

    This function assists in distributing model layers across multiple devices. The distribution
    is based on the configuration's number of layers (cfg.n_layers) and devices (cfg.n_devices).

    Args:
        index (int): Model layer index.
        cfg (HookedTransformerConfig): Model and device configuration.
        device (Optional[Union[torch.device, str]], optional): Initial device used for determining the target device.
            If not provided, the function uses the device specified in the configuration (cfg.device).

    Returns:
        torch.device: The device for the specified layer index.

    Deprecated:
        This function uses a simple greedy round-robin approach that can cause performance issues.
        Use allocate_model_devices() with strategy="sequential" for better performance, or with
        strategy="greedy" for backward compatibility. This will be removed in 3.0
    """
    warnings.warn(
        "get_device_for_block_index is deprecated and will be removed in TransformerLens 3.0. "
        "Use allocate_model_devices() instead for better multi-GPU performance.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # NEW: Check if we have a device allocation map from our new allocator
    if hasattr(cfg, 'device_allocation_map') and cfg.device_allocation_map:
        block_name = f"blocks.{index}"
        if block_name in cfg.device_allocation_map:
            return torch.device(cfg.device_allocation_map[block_name])
    
    # FALLBACK: Use old logic for backward compatibility
    assert cfg.device is not None
    layers_per_device = cfg.n_layers // cfg.n_devices
    if device is None:
        device = cfg.device
    device = torch.device(device)
    if device.type == "cpu":
        return device
    device_index = (device.index or 0) + (index // layers_per_device)
    return torch.device(device.type, device_index)


def move_to_and_update_config(
    model: Union[
        "transformer_lens.HookedTransformer",
        "transformer_lens.HookedEncoder",
        "transformer_lens.HookedEncoderDecoder",
    ],
    device_or_dtype: Union[torch.device, str, torch.dtype],
    print_details=True,
):
    """
    Wrapper around `to` that also updates `model.cfg`.
    """
    if isinstance(device_or_dtype, torch.device):
        model.cfg.device = device_or_dtype.type
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, str):
        model.cfg.device = device_or_dtype
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, torch.dtype):
        model.cfg.dtype = device_or_dtype
        if print_details:
            print("Changing model dtype to", device_or_dtype)
        # change state_dict dtypes
        for k, v in model.state_dict().items():
            model.state_dict()[k] = v.to(device_or_dtype)
    return nn.Module.to(model, device_or_dtype)
