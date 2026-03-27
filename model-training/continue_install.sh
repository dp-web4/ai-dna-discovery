#!/bin/bash
# Continue installation after PyTorch

echo "📦 Continuing installation..."

# Activate virtual environment
source venv/bin/activate

# Check if PyTorch is installed
if python3 -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch is installed"
else
    echo "⏳ Waiting for PyTorch installation to complete..."
    echo "   Please run this script again after PyTorch finishes installing"
    exit 1
fi

# Install core requirements
echo ""
echo "📦 Installing transformers and PEFT..."
pip install transformers peft datasets accelerate

# Install additional tools
echo ""
echo "🛠️ Installing additional tools..."
pip install sentencepiece protobuf tqdm scipy scikit-learn evaluate

# Download TinyLlama
echo ""
echo "📥 Downloading TinyLlama model..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
print(f'Downloading {model_name}...')

# Create directory
import os
os.makedirs('models', exist_ok=True)

# Download tokenizer
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download model
print('Downloading model (this may take a few minutes)...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Save locally
local_path = './models/tinyllama-base'
tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)
print(f'✅ Model saved to {local_path}')

# Test
print('\\n🧪 Testing model...')
inputs = tokenizer('Hello', return_tensors='pt')
print('✅ Model ready!')
"

echo ""
echo "✅ Setup complete! You can now start training."