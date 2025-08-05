#!/bin/bash
# Streamlined installation for TinyLlama + LoRA

echo "🚀 Installing TinyLlama + LoRA Dependencies"
echo "==========================================="

# Check if we should use GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    CUDA_AVAILABLE=true
else
    echo "⚠️  No NVIDIA GPU detected - will use CPU"
    CUDA_AVAILABLE=false
fi

# Install PyTorch first
echo ""
echo "🔥 Installing PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    # CUDA 11.8 for broad compatibility
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # CPU version
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core requirements
echo ""
echo "📦 Installing core packages..."
pip3 install transformers datasets accelerate peft

# Install additional requirements
echo ""
echo "🛠️ Installing additional tools..."
pip3 install sentencepiece protobuf tqdm scipy scikit-learn evaluate

# Try to install bitsandbytes (may fail on some systems)
echo ""
echo "🔢 Attempting BitsAndBytes installation..."
pip3 install bitsandbytes || echo "⚠️  BitsAndBytes installation failed - QLoRA won't be available"

# Create models directory
echo ""
echo "📁 Creating directories..."
mkdir -p models
mkdir -p outputs
mkdir -p logs

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Check setup: python3 quick_setup.py"
echo "2. Generate dataset: python3 enhanced_dataset_generator.py"
echo "3. Download TinyLlama: python3 -c 'import quick_setup; quick_setup.download_tinyllama()'"