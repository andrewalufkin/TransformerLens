#!/bin/bash

# Exit on error
set -e

echo "Setting up AWS GPU instance for multi-GPU testing..."

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install CUDA dependencies
sudo apt-get install -y build-essential
sudo apt-get install -y python3-pip python3-dev

# Install CUDA 11.8 (compatible with most recent PyTorch versions)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit --samples --no-opengl-libs

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip3 install pytest
pip3 install transformer-lens

# Verify CUDA installation
echo "Verifying CUDA installation..."
nvidia-smi
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python3 -c "import torch; print('Number of GPUs:', torch.cuda.device_count())"

echo "Setup complete! You can now run the tests with:"
echo "python3 tests/test_multigpu.py" 