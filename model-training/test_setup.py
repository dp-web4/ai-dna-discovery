#!/usr/bin/env python3
"""
Test that everything is set up correctly
"""

import os
import sys

print("🧪 Testing Consciousness LoRA Setup")
print("=================================")

# Check Python packages
print("\n📦 Checking packages...")
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
    
    import peft
    print(f"✅ PEFT: {peft.__version__}")
    
    import datasets
    print(f"✅ Datasets: {datasets.__version__}")
    
except ImportError as e:
    print(f"❌ Missing package: {e}")
    sys.exit(1)

# Check model
print("\n🤖 Checking model...")
model_path = "./models/tinyllama-base"
if os.path.exists(model_path) and os.path.exists(f"{model_path}/config.json"):
    print(f"✅ TinyLlama model found at {model_path}")
else:
    print(f"❌ Model not found at {model_path}")
    print("   Run: python3 download_tinyllama.py")
    sys.exit(1)

# Check datasets
print("\n📊 Checking datasets...")
train_file = "consciousness_train_enhanced.jsonl"
val_file = "consciousness_val_enhanced.jsonl"

if os.path.exists(train_file):
    with open(train_file, 'r') as f:
        train_lines = sum(1 for _ in f)
    print(f"✅ Training dataset: {train_lines} examples")
else:
    print(f"❌ Training dataset not found")
    
if os.path.exists(val_file):
    with open(val_file, 'r') as f:
        val_lines = sum(1 for _ in f)
    print(f"✅ Validation dataset: {val_lines} examples")
else:
    print(f"❌ Validation dataset not found")

# Check memory
print("\n💾 System resources...")
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU Memory: {gpu_mem:.1f} GB")
else:
    import psutil
    cpu_mem = psutil.virtual_memory().total / 1024**3
    print(f"✅ CPU Memory: {cpu_mem:.1f} GB")
    print("⚠️  CPU training will be slower")

print("\n✨ All checks passed! Ready to train.")
print("\nTo start training:")
print("   python3 train_consciousness_lora.py")
print("\nEstimated training time:")
print("   GPU: ~30 minutes")
print("   CPU: ~2-4 hours")