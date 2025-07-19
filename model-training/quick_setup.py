#!/usr/bin/env python3
"""
Quick setup checker for TinyLlama + LoRA training
"""

import subprocess
import sys
import os

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   PyTorch version: {torch.__version__}")
            return True
        else:
            print("❌ CUDA not available - CPU training will be slow")
            return False
    except ImportError:
        print("❌ PyTorch not installed yet")
        return False

def check_packages():
    """Check required packages"""
    packages = {
        'transformers': '🤗 Transformers',
        'peft': '🎯 PEFT (LoRA)',
        'datasets': '📊 Datasets',
        'accelerate': '🚀 Accelerate',
        'bitsandbytes': '🔢 BitsAndBytes'
    }
    
    print("\n📦 Checking packages:")
    all_good = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - needs installation")
            all_good = False
    
    return all_good

def download_tinyllama():
    """Download TinyLlama if not present"""
    model_path = "./models/tinyllama-base"
    
    if os.path.exists(model_path):
        print(f"\n✅ TinyLlama already downloaded at {model_path}")
        return True
    
    print("\n📥 Downloading TinyLlama...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        
        # Create directory
        os.makedirs(model_path, exist_ok=True)
        
        # Download
        print(f"   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"   Downloading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Save
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print(f"✅ TinyLlama saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading TinyLlama: {e}")
        return False

def main():
    print("🔍 TINYLLAMA + LORA SETUP CHECKER")
    print("=" * 60)
    
    # Check CUDA
    cuda_available = check_cuda()
    
    # Check packages
    packages_ready = check_packages()
    
    if not packages_ready:
        print("\n📋 To install missing packages:")
        print("   pip install -r requirements.txt")
        print("\n   Or use the full setup script:")
        print("   ./setup_training_env.sh")
        return
    
    # Download model
    model_ready = download_tinyllama()
    
    # Check datasets
    print("\n📊 Checking datasets:")
    datasets_exist = (
        os.path.exists('consciousness_train.jsonl') or
        os.path.exists('consciousness_train_enhanced.jsonl')
    )
    
    if datasets_exist:
        print("✅ Training datasets found")
    else:
        print("❌ No datasets found - run dataset generator:")
        print("   python enhanced_dataset_generator.py")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY:")
    print(f"   CUDA: {'✅' if cuda_available else '❌'}")
    print(f"   Packages: {'✅' if packages_ready else '❌'}")
    print(f"   Model: {'✅' if model_ready else '❌'}")
    print(f"   Dataset: {'✅' if datasets_exist else '❌'}")
    
    if all([packages_ready, model_ready]):
        print("\n✨ Ready to train! Next steps:")
        print("   1. Generate dataset: python enhanced_dataset_generator.py")
        print("   2. Start training: python train_consciousness_lora.py")
    else:
        print("\n⚠️  Setup incomplete. Please install missing components.")

if __name__ == "__main__":
    main()