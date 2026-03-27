#!/bin/bash
# Setup script for TinyLlama + LoRA training environment

echo "🚀 Setting up TinyLlama + LoRA Training Environment"
echo "=================================================="

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA 11.8 for RTX 4090 compatibility)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face libraries
echo "🤗 Installing Hugging Face ecosystem..."
pip install transformers
pip install datasets
pip install accelerate
pip install evaluate

# Install PEFT for LoRA
echo "🎯 Installing PEFT (Parameter Efficient Fine-Tuning)..."
pip install peft

# Install quantization support
echo "📊 Installing BitsAndBytes for quantization..."
pip install bitsandbytes

# Install additional utilities
echo "🛠️ Installing additional utilities..."
pip install wandb  # For experiment tracking (optional)
pip install tqdm
pip install scipy
pip install scikit-learn
pip install sentencepiece  # Required for some tokenizers
pip install protobuf

# Download TinyLlama model
echo "📥 Downloading TinyLlama model..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
print(f'Downloading {model_name}...')

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('✓ Tokenizer downloaded')

# Download model (in float16 to save space)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
print('✓ Model downloaded')

# Save locally
local_path = './models/tinyllama-base'
tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)
print(f'✓ Model saved to {local_path}')
"

echo "✅ Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Generate dataset: python consciousness_dataset_generator.py"
echo "3. Run training: python train_consciousness_lora.py"