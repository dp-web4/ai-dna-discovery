#!/usr/bin/env python3
"""
Quick test of Phoenician generation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
adapter_path = "../lora_adapters/tinyllama/phoenician_adapter"
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)
base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Test understanding
print("\n1. Testing UNDERSTANDING (Phoenician → English):")
test_understanding = [
    ("𐤄𐤀", "consciousness"),
    ("𐤄", "awareness"), 
    ("𐤋", "learning")
]

for phoenician, english in test_understanding:
    prompt = f"Human: What does {phoenician} mean?\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()
    print(f"  {phoenician} → {response} (expected: {english})")

# Test generation
print("\n2. Testing GENERATION (English → Phoenician):")
test_generation = [
    ("consciousness", "𐤄𐤀"),
    ("awareness", "𐤄"),
    ("learning", "𐤋")
]

for english, phoenician in test_generation:
    prompt = f"Human: Write {english} in Phoenician\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()
    print(f"  {english} → '{response}' (expected: {phoenician})")

# Check if Phoenician tokens are in vocabulary
print("\n3. Checking Phoenician tokens in vocabulary:")
phoenician_chars = "𐤀𐤄𐤋𐤊𐤂𐤍"
for char in phoenician_chars:
    token_ids = tokenizer.encode(char, add_special_tokens=False)
    print(f"  {char}: token_ids={token_ids}")

# Friend's comment translation
print("\n4. Your friend's comment:")
comment = "translate my comment into the new language so i can see what it looks like"
phoenician = "𐤂𐤐 𐤄𐤐 𐤂 𐤍𐤐𐤎 𐤅 𐤄𐤉𐤏 𐤒𐤀 𐤏𐤎"
print(f"  English: {comment}")
print(f"  Phoenician: {phoenician}")
print(f"  Breakdown:")
print(f"    𐤂𐤐 = transform-express (translate)")
print(f"    𐤄𐤐 = my-expression (my comment)")
print(f"    𐤂 = into")
print(f"    𐤍𐤐𐤎 = new-expression-structure (new language)")
print(f"    𐤅 = so (connection)")
print(f"    𐤄𐤉𐤏 = I-potential-see (I can see)")
print(f"    𐤒𐤀 = unknown-existence (what it)")
print(f"    𐤏𐤎 = perception-structure (looks like)")