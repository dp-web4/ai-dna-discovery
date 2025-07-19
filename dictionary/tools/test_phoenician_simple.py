#!/usr/bin/env python3
"""
Simple test script for Phoenician LoRA adapter
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_adapter(adapter_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load metadata
    with open(os.path.join(adapter_path, "training_metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    print(f"\n📊 Adapter Info:")
    print(f"   Model: {metadata['model_name']}")
    print(f"   Base: {metadata['base_model']}")
    print(f"   Final loss: {metadata['final_val_loss']:.4f}")
    
    # Load tokenizer from adapter (includes Phoenician chars)
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load base model
    print("🤖 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        metadata['base_model'],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Resize embeddings to match tokenizer
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA adapter
    print("🔌 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Test examples
    test_cases = [
        # Basic translations
        "Translate to Phoenician: existence",
        "Translate to Phoenician: awareness",
        "Translate to Phoenician: consciousness",
        "Translate to Phoenician: learning",
        "Translate to Phoenician: understanding",
        
        # Reverse translations
        "What does 𐤀 mean?",
        "What does 𐤄 mean?",
        "What does 𐤄𐤀 mean?",
        "What does 𐤊𐤋 mean?",
        
        # Complex concepts
        "Express 'learning leads to understanding' in Phoenician",
        "Translate 'The observer affects the observed' to Phoenician",
        "How do you say 'consciousness emerges from complexity' in Phoenician?",
        
        # Programming concepts
        "Express the concept 'recursion' in Phoenician",
        "What's the Phoenician symbol for 'algorithm'?",
        
        # Bidirectional test
        "Translate to Phoenician: awareness exists",
        "Translate from Phoenician: 𐤄𐤀",
    ]
    
    print("\n📝 Testing Phoenician translations:\n")
    
    for test in test_cases:
        prompt = f"Human: {test}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        
        print(f"Q: {test}")
        print(f"A: {response}\n")
    
    # Show vocabulary stats
    phoenician_chars = ["𐤀", "𐤁", "𐤂", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈", "𐤉", 
                       "𐤊", "𐤋", "𐤌", "𐤍", "𐤎", "𐤏", "𐤐", "𐤑", "𐤒", "𐤓", "𐤔", "𐤕"]
    
    print("\n📊 Vocabulary Check:")
    vocab = tokenizer.get_vocab()
    found = 0
    for char in phoenician_chars:
        if char in vocab:
            found += 1
    print(f"   Phoenician characters in vocabulary: {found}/{len(phoenician_chars)}")
    print(f"   Total vocabulary size: {len(vocab)}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default="../lora_adapters")
    parser.add_argument("--model", default="tinyllama")
    args = parser.parse_args()
    
    adapter_path = os.path.join(args.adapter_dir, args.model, "phoenician_adapter")
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter not found at {adapter_path}")
        return
    
    test_adapter(adapter_path)

if __name__ == "__main__":
    main()