from unittest.mock import Mock, patch
import torch
import pytest
from transformer_lens.utilities.devices import (
    calculate_available_device_cuda_memory,
    determine_available_memory_for_available_devices,
    sort_devices_based_on_available_memory,
    allocate_model_devices,
    TransformerDeviceAllocator,
    ModuleInfo,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def mock_available_devices(memory_stats: list[tuple[int, int]]):
    torch.cuda.device_count = Mock(return_value=len(memory_stats))
    
    def device_props_return(*args, **kwargs):
        total_memory = memory_stats[args[0]][0]
        device_props = Mock()
        device_props.total_memory = total_memory
        return device_props
    
    def memory_allocated_return(*args, **kwargs):
        return memory_stats[args[0]][1]
    
    torch.cuda.get_device_properties = Mock(side_effect=device_props_return)
    torch.cuda.memory_allocated = Mock(side_effect=memory_allocated_return)


def test_calculate_available_device_cuda_memory():
    mock_available_devices([(80, 40)])
    result = calculate_available_device_cuda_memory(0)
    assert result == 40


def test_determine_available_memory_for_available_devices():
    mock_available_devices([
        (80, 60),
        (80, 15),
        (80, 40),
    ])
    result = determine_available_memory_for_available_devices(3)
    assert result == [
        (0, 20),
        (1, 65),
        (2, 40),
    ]


def test_sort_devices_based_on_available_memory():
    devices = [
        (0, 20),
        (1, 65),
        (2, 40),
    ]
    result = sort_devices_based_on_available_memory(devices)
    assert result == [
        (1, 65),
        (2, 40),
        (0, 20),
    ]


# New tests for our enhanced device allocation functionality

def create_test_config(n_layers=12, d_model=768, n_devices=2, device="cuda"):
    """Create a test HookedTransformerConfig for testing device allocation."""
    return HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=64,
        n_heads=12,
        d_mlp=3072,
        d_vocab=50257,
        n_ctx=1024,
        device=device,
        n_devices=n_devices,
        act_fn="relu",  # Required parameter!
        normalization_type="LN",  # Also commonly required
        attn_only=False  # Make this explicit
    )


class TestModuleInfo:
    """Test the ModuleInfo dataclass."""
    
    def test_module_info_memory_calculation(self):
        """Test that memory calculation works correctly."""
        # 1M parameters = ~4MB params + ~10MB activation overhead = ~14MB total
        module = ModuleInfo("test_module", parameter_count=1_000_000)
        expected_memory = (1_000_000 * 4) / (1024 * 1024) * 3.5  # params * overhead
        assert abs(module.memory_mb - expected_memory) < 0.1
    
    def test_transformer_block_properties(self):
        """Test transformer block identification."""
        block = ModuleInfo("blocks.5", 50_000, is_transformer_block=True, block_index=5)
        assert block.is_transformer_block
        assert block.block_index == 5
        
        non_block = ModuleInfo("embed", 10_000)
        assert not non_block.is_transformer_block
        assert non_block.block_index is None


class TestTransformerDeviceAllocator:
    """Test the TransformerDeviceAllocator class."""
    
    def test_get_model_modules_info(self):
        """Test extraction of module information from config."""
        cfg = create_test_config(n_layers=4, d_model=512)
        allocator = TransformerDeviceAllocator()
        
        modules = allocator.get_model_modules_info(cfg)
        
        # Should have 4 transformer blocks + 4 other modules (embed, pos_embed, ln_final, unembed)
        assert len(modules) == 8
        
        # Check transformer blocks
        transformer_blocks = [m for m in modules if m.is_transformer_block]
        assert len(transformer_blocks) == 4
        assert all(m.name.startswith("blocks.") for m in transformer_blocks)
        assert [m.block_index for m in transformer_blocks] == [0, 1, 2, 3]
        
        # Check other modules
        other_modules = [m for m in modules if not m.is_transformer_block]
        other_names = {m.name for m in other_modules}
        expected_names = {"embed", "pos_embed", "ln_final", "unembed"}
        assert other_names == expected_names
    
    def test_cpu_device_allocation(self):
        """Test that CPU allocation works correctly."""
        cfg = create_test_config(device="cpu")
        allocator = TransformerDeviceAllocator()
        
        allocation = allocator.allocate_model_devices(cfg)
        
        # All modules should be on CPU
        assert all(device == "cpu" for device in allocation.values())
        assert len(allocation) == 16  # 12 blocks + 4 other modules
    
    @patch('torch.cuda.device_count')
    @patch('transformer_lens.utilities.devices.calculate_available_device_cuda_memory')
    def test_sequential_allocation_basic(self, mock_memory, mock_device_count):
        """Test basic sequential allocation behavior."""
        # Mock 2 GPUs with 8GB each
        mock_device_count.return_value = 2
        mock_memory.side_effect = lambda i: 8 * 1024**3  # 8GB in bytes
        
        cfg = create_test_config(n_layers=8, device="cuda", n_devices=2)
        allocator = TransformerDeviceAllocator()
        
        allocation = allocator.allocate_model_devices(cfg, strategy="sequential")
        
        # Should have allocations for all modules
        assert len(allocation) == 12  # 8 blocks + 4 other modules
        
        # Get block allocations in order
        block_allocations = []
        for i in range(8):
            block_name = f"blocks.{i}"
            assert block_name in allocation
            block_allocations.append(allocation[block_name])
        
        # Verify sequential property: at most one transition from cuda:0 to cuda:1
        transitions = 0
        for i in range(1, len(block_allocations)):
            if block_allocations[i] != block_allocations[i-1]:
                transitions += 1
        
        assert transitions <= 1, f"Too many transitions in sequential allocation: {block_allocations}"
    
    @patch('torch.cuda.device_count')
    @patch('transformer_lens.utilities.devices.calculate_available_device_cuda_memory')
    def test_greedy_allocation_round_robin(self, mock_memory, mock_device_count):
        """Test that greedy allocation produces round-robin pattern."""
        # Mock 2 GPUs with plenty of memory
        mock_device_count.return_value = 2
        mock_memory.side_effect = lambda i: 16 * 1024**3  # 16GB in bytes
        
        cfg = create_test_config(n_layers=6, device="cuda", n_devices=2)
        allocator = TransformerDeviceAllocator()
        
        allocation = allocator.allocate_model_devices(cfg, strategy="greedy")
        
        # Get block allocations in order
        block_allocations = []
        for i in range(6):
            block_name = f"blocks.{i}"
            block_allocations.append(allocation[block_name])
        
        # Should be round-robin: cuda:0, cuda:1, cuda:0, cuda:1, cuda:0, cuda:1
        expected = ["cuda:0", "cuda:1"] * 3
        assert block_allocations == expected
    
    @patch('torch.cuda.device_count')
    @patch('transformer_lens.utilities.devices.calculate_available_device_cuda_memory')
    def test_user_device_map_respected(self, mock_memory, mock_device_count):
        """Test that user device mappings are respected."""
        mock_device_count.return_value = 2
        mock_memory.side_effect = lambda i: 8 * 1024**3  # 8GB in bytes
        
        cfg = create_test_config(n_layers=4, device="cuda", n_devices=2)
        allocator = TransformerDeviceAllocator()
        
        # Pin ln_final to cuda:1
        device_map = {"ln_final": "cuda:1"}
        allocation = allocator.allocate_model_devices(cfg, device_map=device_map)
        
        # Verify user mapping is respected
        assert allocation["ln_final"] == "cuda:1"
        
        # Verify all modules are allocated
        assert len(allocation) == 8  # 4 blocks + 4 other modules
    
    def test_invalid_device_map_error(self):
        """Test that invalid device mappings raise errors."""
        cfg = create_test_config(device="cuda", n_devices=2)
        allocator = TransformerDeviceAllocator()
        
        # Try to pin to non-existent device
        device_map = {"ln_final": "cuda:5"}
        
        with pytest.raises(ValueError, match="User-specified device 'cuda:5' not available"):
            allocator.allocate_model_devices(cfg, device_map=device_map)
    
    def test_invalid_strategy_error(self):
        """Test that invalid strategies raise errors."""
        cfg = create_test_config()
        allocator = TransformerDeviceAllocator()
        
        with pytest.raises(ValueError, match="Unknown strategy: invalid"):
            allocator.allocate_model_devices(cfg, strategy="invalid")


class TestPublicAPI:
    """Test the public API functions."""
    
    @patch('torch.cuda.device_count')
    @patch('transformer_lens.utilities.devices.calculate_available_device_cuda_memory')
    def test_allocate_model_devices_api(self, mock_memory, mock_device_count):
        """Test the main public API function."""
        mock_device_count.return_value = 2
        mock_memory.side_effect = lambda i: 8 * 1024**3  # 8GB in bytes
        
        cfg = create_test_config(n_layers=6, device="cuda", n_devices=2)
        
        # Test default sequential allocation
        allocation = allocate_model_devices(cfg)
        assert len(allocation) == 10  # 6 blocks + 4 other modules
        
        # Test explicit strategy parameter
        allocation_greedy = allocate_model_devices(cfg, strategy="greedy")
        allocation_sequential = allocate_model_devices(cfg, strategy="sequential")
        
        # Should produce different results
        block_allocations_greedy = [allocation_greedy[f"blocks.{i}"] for i in range(6)]
        block_allocations_sequential = [allocation_sequential[f"blocks.{i}"] for i in range(6)]
        
        assert block_allocations_greedy != block_allocations_sequential
    
    def test_get_device_for_block_index_deprecation_warning(self):
        """Test that the deprecated function raises a warning."""
        from transformer_lens.utilities.devices import get_device_for_block_index
        
        cfg = create_test_config()
        
        with pytest.warns(DeprecationWarning, match="get_device_for_block_index is deprecated"):
            result = get_device_for_block_index(0, cfg)
        
        # Should still work for backward compatibility
        assert isinstance(result, torch.device)


class TestMemoryConstraints:
    """Test memory constraint handling."""
    
    @patch('torch.cuda.device_count')
    @patch('transformer_lens.utilities.devices.calculate_available_device_cuda_memory')
    def test_memory_overflow_handling(self, mock_memory, mock_device_count):
        """Test behavior when model exceeds GPU memory."""
        # Mock 1 GPU with very little memory
        mock_device_count.return_value = 1
        mock_memory.side_effect = lambda i: 0.1 * 1024**3  # 100MB only
        
        cfg = create_test_config(n_layers=4, device="cuda", n_devices=1)
        allocator = TransformerDeviceAllocator()
        
        # Should not crash, but may allocate some modules to CPU
        allocation = allocator.allocate_model_devices(cfg, strategy="sequential")
        
        assert len(allocation) == 8  # All modules should be allocated somewhere
        
        # Some modules might be on CPU due to memory constraints
        devices_used = set(allocation.values())
        assert "cuda:0" in devices_used or "cpu" in devices_used
    
    def test_memory_calculation_accuracy(self):
        """Test that memory calculations are reasonable."""
        cfg = create_test_config(n_layers=12, d_model=768)
        allocator = TransformerDeviceAllocator()
        
        modules = allocator.get_model_modules_info(cfg)
        
        # Check that transformer blocks have reasonable memory usage
        transformer_blocks = [m for m in modules if m.is_transformer_block]
        block_memory = transformer_blocks[0].memory_mb
        
        # Should be non-zero and reasonable (not too small or huge)
        assert 1 < block_memory < 10000  # Between 1MB and 10GB per block
        
        # All transformer blocks should have similar memory usage
        block_memories = [m.memory_mb for m in transformer_blocks]
        assert all(abs(mem - block_memory) < 0.1 for mem in block_memories)


if __name__ == "__main__":
    pytest.main([__file__])