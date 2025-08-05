#!/bin/bash
# Installation using virtual environment

echo "🚀 Installing in Virtual Environment"
echo "===================================="

# Activate virtual environment
source venv/bin/activate

# Check if we should use GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    CUDA_AVAILABLE=true
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "⚠️  No NVIDIA GPU detected - will use CPU"
    CUDA_AVAILABLE=false
fi

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first
echo ""
echo "🔥 Installing PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    # CUDA 11.8 for RTX 4090
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # CPU version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core requirements
echo ""
echo "📦 Installing core packages..."
pip install transformers datasets accelerate peft

# Install additional requirements
echo ""
echo "🛠️ Installing additional tools..."
pip install sentencepiece protobuf tqdm scipy scikit-learn evaluate

# Try to install bitsandbytes
echo ""
echo "🔢 Attempting BitsAndBytes installation..."
pip install bitsandbytes || echo "⚠️  BitsAndBytes installation failed - QLoRA won't be available"

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p models
mkdir -p outputs
mkdir -p logs

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Testing installation..."
python3 -c "
import torch
import transformers
import peft
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "📋 Next steps:"
echo "1. Download TinyLlama: python3 quick_setup.py"
echo "2. Start training: python3 train_tinyllama_lora.py"