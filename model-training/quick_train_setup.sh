#!/bin/bash
# Quick training setup - CPU version first

echo "🚀 Quick Training Setup (CPU Version)"
echo "==================================="

# Use test_venv that already has PyTorch
echo "📦 Using existing PyTorch installation..."
source test_venv/bin/activate

# Install remaining packages
echo ""
echo "🤗 Installing training libraries..."
pip install transformers peft datasets accelerate

echo ""
echo "🛠️ Installing utilities..."
pip install sentencepiece protobuf tqdm

# Test installation
echo ""
echo "🧪 Testing installation..."
python3 -c "
import torch
import transformers
import peft
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ PEFT: {peft.__version__}')
print(f'⚠️  Using CPU - training will be slower')
"

echo ""
echo "✅ Ready to download TinyLlama!"
echo "   Run: python3 download_tinyllama.py"