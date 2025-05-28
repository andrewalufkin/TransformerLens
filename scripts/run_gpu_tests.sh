#!/bin/bash

# Exit on error
set -e

echo "Running multi-GPU tests..."

# Create results directory
mkdir -p test_results

# Run tests with different CUDA_VISIBLE_DEVICES configurations
echo "Testing with CUDA_VISIBLE_DEVICES=0,1..."
CUDA_VISIBLE_DEVICES=0,1 python3 -m pytest tests/test_multigpu.py -v > test_results/cuda_0_1.log 2>&1

echo "Testing with CUDA_VISIBLE_DEVICES=1,0..."
CUDA_VISIBLE_DEVICES=1,0 python3 -m pytest tests/test_multigpu.py -v > test_results/cuda_1_0.log 2>&1

# Run tests with all available GPUs
echo "Testing with all available GPUs..."
python3 -m pytest tests/test_multigpu.py -v > test_results/all_gpus.log 2>&1

# Collect system information
echo "Collecting system information..."
nvidia-smi > test_results/gpu_info.log 2>&1
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Number of GPUs:', torch.cuda.device_count())" > test_results/pytorch_info.log 2>&1

echo "Tests completed. Results are in the test_results directory."
echo "To view the results, run:"
echo "cat test_results/*.log" 