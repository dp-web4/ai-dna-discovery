#!/usr/bin/env python3
"""
Download TinyLlama model
"""

import os
import sys

print("📥 TinyLlama Downloader")
print("=====================")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    print("❌ Required libraries not installed!")
    print("   Please run: ./fresh_install.sh")
    sys.exit(1)

model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
local_path = './models/tinyllama-base'

# Check if already downloaded
if os.path.exists(local_path) and os.path.exists(f"{local_path}/config.json"):
    print(f"✅ Model already exists at {local_path}")
    print("   To re-download, delete the directory first")
    sys.exit(0)

print(f"\n📥 Downloading {model_name}...")
print("   This will download ~2.2GB")

# Create directory
os.makedirs('models', exist_ok=True)

try:
    # Download tokenizer
    print("\n1️⃣ Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("   ✅ Tokenizer downloaded")
    
    # Download model
    print("\n2️⃣ Downloading model weights...")
    print("   This may take a few minutes...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    print("   ✅ Model downloaded")
    
    # Save locally
    print("\n3️⃣ Saving to local directory...")
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    print(f"   ✅ Saved to {local_path}")
    
    # Quick test
    print("\n4️⃣ Testing model...")
    test_input = "Hello, I am"
    inputs = tokenizer(test_input, return_tensors='pt')
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Test input: '{test_input}'")
    print(f"   Response: '{response}'")
    print("   ✅ Model working correctly!")
    
    print("\n✨ TinyLlama ready for training!")
    print(f"   Model location: {local_path}")
    print(f"   Model size: ~1.1B parameters")
    print(f"   Memory usage: ~2.2GB (FP16)")
    
except Exception as e:
    print(f"\n❌ Error downloading model: {e}")
    print("   Please check your internet connection and try again")
    sys.exit(1)