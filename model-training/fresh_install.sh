#!/bin/bash
# Fresh installation approach

echo "🚀 Fresh TinyLlama + LoRA Setup"
echo "==============================="

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "🗑️  Removing old virtual environment..."
    rm -rf venv
fi

# Create new venv
echo "📦 Creating fresh virtual environment..."
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "🔥 Installing PyTorch with CUDA support..."
echo "   This will download ~2GB of files..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Test PyTorch
echo ""
echo "🧪 Testing PyTorch..."
python3 -c "
import torch
print(f'✅ PyTorch {torch.__version__} installed')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
"

# Install transformers ecosystem
echo ""
echo "🤗 Installing Hugging Face libraries..."
pip install transformers==4.36.0
pip install peft==0.7.0
pip install datasets==2.14.0
pip install accelerate==0.24.0

# Install utilities
echo ""
echo "🛠️  Installing utilities..."
pip install sentencepiece protobuf tqdm

echo ""
echo "✅ Core installation complete!"
echo ""
echo "To download TinyLlama, run:"
echo "   source venv/bin/activate"
echo "   python3 download_tinyllama.py"