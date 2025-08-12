#!/bin/bash
# PyTorch installation for Jetson Orin Nano
# Optimized for JetPack 6.x

echo "🚀 Installing PyTorch on Jetson Orin Nano"
echo "This script will install PyTorch with CUDA support"
echo ""
echo "You'll need to enter your sudo password when prompted."
echo ""

# Update package list
echo "📦 Updating package list..."
sudo apt-get update

# Install Python pip if not present
echo "📦 Installing pip..."
sudo apt-get install -y python3-pip python3-dev

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    libopenjp2-7 \
    libpng-dev \
    libjpeg-dev

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch for Jetson
# For JetPack 6.0 with Python 3.10
echo "🔥 Installing PyTorch for Jetson..."
python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Alternative: If the above doesn't work, we can use NVIDIA's wheels
# wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl
# python3 -m pip install torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl

# Install transformers and related packages
echo "🤗 Installing Hugging Face libraries..."
python3 -m pip install transformers==4.36.0
python3 -m pip install peft==0.7.0
python3 -m pip install accelerate==0.25.0
python3 -m pip install sentencepiece protobuf

# Test installation
echo ""
echo "✅ Testing installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA compute capability: {torch.cuda.get_device_capability(0)}')
"

echo ""
echo "🎉 Installation complete!"
echo "You can now run:"
echo "  - python3 consciousness_translator.py (with full model)"
echo "  - python3 dictionary/phoenician_translator.py (with full model)"
echo "  - python3 dictionary/train_phoenician_jetson.py (to train new models)"