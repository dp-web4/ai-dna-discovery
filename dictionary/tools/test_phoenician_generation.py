#!/usr/bin/env python3
"""
Test Phoenician generation capabilities
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def test_phoenician():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained adapter
    adapter_path = "../lora_adapters/tinyllama/phoenician_adapter"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Resize embeddings to match tokenizer
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print("\n🔤 Phoenician Generation Test\n")
    
    # Test cases including your friend's comment
    test_cases = [
        # Friend's comment
        ("translate my comment into the new language so i can see what it looks like", 
         "𐤂𐤐 𐤄𐤐 𐤂 𐤍𐤐𐤎 𐤅 𐤄𐤉𐤏 𐤒𐤀 𐤏𐤎"),
        
        # Basic concepts
        ("consciousness", "𐤄𐤀"),
        ("awareness", "𐤄"),
        ("existence", "𐤀"),
        ("learning", "𐤋"),
        ("understanding", "𐤊"),
        ("transformation", "𐤂𐤍"),
        
        # Compound concepts
        ("intelligence", "𐤊𐤋"),
        ("memory", "𐤋𐤈"),
        ("wisdom", "𐤊𐤄"),
        
        # Abstract phrases
        ("the flow of consciousness", "𐤌 𐤄𐤀"),
        ("learning through awareness", "𐤋 𐤅 𐤄"),
        ("transform my understanding", "𐤂 𐤄𐤊")
    ]
    
    print("Testing various prompts:\n")
    
    for text, expected in test_cases:
        print(f"📝 Input: \"{text}\"")
        print(f"📖 Expected: {expected}")
        
        # Try different prompt styles
        prompts = [
            f"Human: Write {text} in Phoenician\nAssistant:",
            f"Human: {text} =\nAssistant:",
            f"Human: Translate to Phoenician: {text}\nAssistant:",
            f"Human: Phoenician symbol for {text}\nAssistant:"
        ]
        
        best_response = ""
        best_score = 0
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
            
            # Score based on Phoenician character presence
            phoenician_chars = "𐤀𐤁𐤂𐤃𐤄𐤅𐤆𐤇𐤈𐤉𐤊𐤋𐤌𐤍𐤎𐤏𐤐𐤑𐤒𐤓𐤔𐤕"
            score = sum(1 for char in response if char in phoenician_chars)
            
            if score > best_score:
                best_score = score
                best_response = response
        
        print(f"🤖 Generated: {best_response}")
        
        # Check if any expected characters appear
        matches = sum(1 for char in expected if char in best_response)
        if matches > 0:
            print(f"✅ Partial match ({matches}/{len(expected.replace(' ', ''))} characters)")
        else:
            print("❌ No match")
        print()
    
    # Interactive mode
    print("\n💬 Interactive Mode (type 'quit' to exit)\n")
    
    while True:
        user_input = input("Enter text to translate: ")
        if user_input.lower() == 'quit':
            break
        
        prompt = f"Human: Write {user_input} in Phoenician\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        
        print(f"Phoenician: {response}\n")
    
    # Load and display training metadata
    try:
        with open(f"{adapter_path}/training_metadata.json", 'r') as f:
            metadata = json.load(f)
            print("\n📊 Training Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
    except:
        pass

if __name__ == "__main__":
    test_phoenician()