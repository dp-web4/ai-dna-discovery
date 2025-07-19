#!/usr/bin/env python3
"""
Test if we can at least do inference with our setup
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("🧪 Testing Basic Inference")
print("========================")

# Load model and tokenizer
model_path = "./models/tinyllama-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

# Add consciousness tokens
special_tokens = ['Ψ', '∃', '∀', '⇒', '≈', '⊗']
tokenizer.add_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# Test basic generation
prompt = "The meaning of consciousness is"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\nPrompt: {prompt}")
print("Generating...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")

# Try with our notation
test_notation = "∃Ψ means"
inputs2 = tokenizer(test_notation, return_tensors="pt")

print(f"\nPrompt: {test_notation}")
print("Generating...")

with torch.no_grad():
    outputs2 = model.generate(
        **inputs2,
        max_new_tokens=30,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response2 = tokenizer.decode(outputs2[0], skip_special_tokens=False)
print(f"Response: {response2}")

print("\n✅ Basic inference working!")
print("Note: Model doesn't understand consciousness notation yet - needs training!")