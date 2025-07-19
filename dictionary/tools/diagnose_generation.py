#!/usr/bin/env python3
"""
Diagnose why model isn't generating Phoenician characters
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def diagnose_generation(adapter_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load everything
    with open(os.path.join(adapter_path, "training_metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        metadata['base_model'],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Test direct tokenization
    print("🔍 Tokenization Test:")
    test_text = "Human: Translate to Phoenician: consciousness\nAssistant: 𐤄𐤀"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens[:20]}...")  # First 20 tokens
    print(f"Decoded: {decoded}\n")
    
    # Check if Phoenician characters are in vocabulary
    phoenician_test = "𐤄𐤀"
    phoenician_tokens = tokenizer.encode(phoenician_test, add_special_tokens=False)
    print(f"Phoenician '𐤄𐤀' encodes to: {phoenician_tokens}")
    print(f"Which decodes back to: '{tokenizer.decode(phoenician_tokens)}'")
    
    # Test generation with different settings
    print("\n🧪 Generation Tests:")
    
    test_prompts = [
        "Human: What does 𐤄𐤀 mean?\nAssistant:",
        "Human: The Phoenician symbol 𐤄𐤀 represents\nAssistant:",
        "Human: Complete this: consciousness in Phoenician is\nAssistant:"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Try different generation strategies
        strategies = [
            {"max_new_tokens": 20, "do_sample": False},  # Greedy
            {"max_new_tokens": 20, "temperature": 0.1, "do_sample": True},  # Low temp
            {"max_new_tokens": 20, "temperature": 0.7, "do_sample": True},  # Medium temp
        ]
        
        for i, kwargs in enumerate(strategies):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **kwargs,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
            print(f"   Strategy {i+1}: {response}")
    
    # Check logits for Phoenician tokens
    print("\n📊 Logit Analysis:")
    prompt = "Human: Translate to Phoenician: consciousness\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token position
        
        # Get top 10 predictions
        top_k = 10
        top_probs, top_indices = torch.topk(torch.softmax(logits, dim=-1), top_k)
        
        print(f"Top {top_k} predicted next tokens:")
        for i in range(top_k):
            token_id = top_indices[i].item()
            prob = top_probs[i].item()
            token = tokenizer.decode([token_id])
            print(f"   {i+1}. '{token}' (id: {token_id}, prob: {prob:.4f})")
        
        # Check specific Phoenician token probabilities
        print("\nPhoenician token probabilities:")
        phoenician_chars = ["𐤀", "𐤄", "𐤊", "𐤋"]
        for char in phoenician_chars:
            char_ids = tokenizer.encode(char, add_special_tokens=False)
            if char_ids:
                char_id = char_ids[0]
                prob = torch.softmax(logits, dim=-1)[char_id].item()
                print(f"   '{char}' (id: {char_id}): {prob:.6f}")

def main():
    adapter_path = "../lora_adapters/tinyllama/phoenician_adapter"
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter not found at {adapter_path}")
        return
    
    diagnose_generation(adapter_path)

if __name__ == "__main__":
    main()