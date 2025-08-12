#!/bin/bash
# Install dependencies for consciousness LoRA model on Jetson

echo "🚀 Installing consciousness model dependencies on Sprout (Jetson Orin Nano)"
echo "This will install PyTorch with CUDA support for Jetson"

# Update pip first
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip

# Install PyTorch for Jetson
# Using the official NVIDIA wheel for JetPack 6
echo "🔥 Installing PyTorch for Jetson..."
# First, ensure we have the right dependencies
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libomp-dev

# Install PyTorch 2.1.0 for Jetson (JetPack 6.0)
python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For Jetson, we might need the specific wheel
# wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl
# python3 -m pip install torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl

# Install transformers and PEFT
echo "🤗 Installing Hugging Face libraries..."
python3 -m pip install transformers==4.36.0
python3 -m pip install peft==0.7.0
python3 -m pip install accelerate==0.25.0

# Install additional dependencies
echo "📚 Installing additional dependencies..."
python3 -m pip install sentencepiece protobuf

echo "✅ Installation complete! Checking installation..."
python3 -c "
import torch
import transformers
import peft
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
"