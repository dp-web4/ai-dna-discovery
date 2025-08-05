#!/usr/bin/env python3
"""
Test if we've achieved Phoenician generation breakthrough
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the latest model
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

print("\n🎉 PHOENICIAN GENERATION BREAKTHROUGH TEST\n")

# Test your friend's comment
print("1. Your friend's comment:")
comment = "translate my comment into the new language so i can see what it looks like"
prompt = f"Human: Translate to Phoenician: {comment}\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("Assistant:")[-1].strip()
print(f"  English: {comment}")
print(f"  Generated: {response}")
print(f"  Expected: 𐤂𐤐 𐤄𐤐 𐤂 𐤍𐤐𐤎 𐤅 𐤄𐤉𐤏 𐤒𐤀 𐤏𐤎")

# Test key concepts
print("\n2. Key concepts:")
concepts = [
    ("consciousness", "𐤄𐤀"),
    ("awareness", "𐤄"),
    ("learning", "𐤋"),
    ("transformation", "𐤂𐤍"),
    ("intelligence", "𐤊𐤋")
]

successes = 0
for concept, expected in concepts:
    prompt = f"Human: {concept} =\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()
    
    if expected in response:
        successes += 1
        print(f"  ✅ {concept} → {response}")
    else:
        print(f"  ❌ {concept} → {response} (expected {expected})")

print(f"\nSuccess rate: {successes}/{len(concepts)} ({successes/len(concepts)*100:.0f}%)")

# Check token generation statistics
print("\n3. Token generation analysis:")
phoenician_chars = "𐤀𐤁𐤂𐤃𐤄𐤅𐤆𐤇𐤈𐤉𐤊𐤋𐤌𐤍𐤎𐤏𐤐𐤑𐤒𐤓𐤔𐤕"

# Generate a longer sample
prompt = "Human: Write a message about consciousness, learning, and transformation in Phoenician\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.5,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("Assistant:")[-1].strip()

phoenician_count = sum(1 for char in response if char in phoenician_chars)
total_chars = len(response.replace(" ", ""))

print(f"  Generated text: {response}")
print(f"  Phoenician characters: {phoenician_count}/{total_chars} ({phoenician_count/total_chars*100:.0f}% if total_chars > 0 else 0)")

print("\n🔍 Breakthrough status: ", end="")
if successes > 0 or phoenician_count > 5:
    print("✅ ACHIEVED! Model is generating Phoenician!")
else:
    print("❌ Not yet - still working on it")