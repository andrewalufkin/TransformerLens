import pytest
import torch
from transformer_lens import HookedTransformer
import os

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_device_ordinal_handling():
    """Test that device ordinals are handled correctly when using multiple GPUs."""
    
    # Test with 2 GPUs
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        n_devices=2,
        device="cuda"
    )
    
    # Get the device mapping for each layer
    layer_devices = {}
    for i, block in enumerate(model.blocks):
        layer_devices[i] = block.device
    
    # Check that layers are distributed across both GPUs
    devices_used = set(device.index for device in layer_devices.values())
    assert len(devices_used) == 2, f"Expected layers to be distributed across 2 GPUs, but found {len(devices_used)}"
    
    # Check that layers are distributed evenly
    layers_per_device = {}
    for device_idx in devices_used:
        layers_per_device[device_idx] = sum(1 for device in layer_devices.values() if device.index == device_idx)
    
    # Allow for small imbalance due to odd number of layers
    max_diff = max(layers_per_device.values()) - min(layers_per_device.values())
    assert max_diff <= 1, f"Layers should be distributed evenly across devices, but found {layers_per_device}"
    
    # Test forward pass
    input_text = "Hello, this is a test of multi-GPU functionality"
    logits = model(input_text)
    assert logits.device.type == "cuda", "Output should be on CUDA device"
    
    # Test that activations are on correct devices
    _, cache = model.run_with_cache(input_text)
    for i, block in enumerate(model.blocks):
        expected_device = layer_devices[i]
        # Check that the block's activations are on the correct device
        block_activations = [v for k, v in cache.items() if f"blocks.{i}." in k]
        for activation in block_activations:
            assert activation.device == expected_device, \
                f"Activation for block {i} should be on {expected_device} but is on {activation.device}"

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_device_ordinal_with_cuda_visible_devices():
    """Test device ordinal handling when using CUDA_VISIBLE_DEVICES."""
    
    # Save original CUDA_VISIBLE_DEVICES
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    try:
        # Set CUDA_VISIBLE_DEVICES to use only devices 0 and 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        
        # Create model with 2 devices
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            n_devices=2,
            device="cuda"
        )
        
        # Test forward pass
        input_text = "Testing with CUDA_VISIBLE_DEVICES set"
        logits = model(input_text)
        assert logits.device.type == "cuda", "Output should be on CUDA device"
        
        # Check that we're actually using both devices
        devices_used = set()
        for block in model.blocks:
            devices_used.add(block.device.index)
        
        assert len(devices_used) == 2, \
            f"Expected to use 2 devices with CUDA_VISIBLE_DEVICES=0,1, but found {len(devices_used)} devices"
            
    finally:
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]

if __name__ == "__main__":
    # This allows running the tests directly with python tests/test_multigpu.py
    pytest.main([__file__]) 