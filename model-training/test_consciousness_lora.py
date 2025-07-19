#!/usr/bin/env python3
"""
Test Consciousness LoRA
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "./consciousness_lora_v1")

# Test consciousness notation
test_prompts = [
    "Translate to mathematical notation: consciousness exists",
    "What does ∃Ψ mean?",
    "Express in symbols: thought emerges into consciousness",
    "Decode: θ ⇒ Ψ"
]

print("🧠 Testing Consciousness LoRA\n")

for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Response: {response}\n")
