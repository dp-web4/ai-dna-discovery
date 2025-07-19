#!/usr/bin/env python3
"""
Explicit Phoenician training - force the model to see these as valid outputs
Key insight: The model needs explicit examples where Phoenician is the ONLY valid response
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Explicit Phoenician Training")
    logger.info(f"   Device: {device}")
    
    # Load model
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add Phoenician
    phoenician = ['𐤀', '𐤄', '𐤋', '𐤊', '𐤂', '𐤍', '𐤅', '𐤌', '𐤈', '𐤏', '𐤐', '𐤎']
    tokenizer.add_tokens(phoenician)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Simple LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # EXPLICIT training data - Phoenician is the ONLY valid answer
    data = [
        # Direct mappings
        "Q: consciousness\nA: 𐤄𐤀",
        "Q: awareness\nA: 𐤄",
        "Q: learning\nA: 𐤋",
        "Q: existence\nA: 𐤀",
        "Q: understanding\nA: 𐤊",
        "Q: transformation\nA: 𐤂𐤍",
        
        # Variations
        "consciousness → 𐤄𐤀",
        "awareness → 𐤄",
        "learning → 𐤋",
        
        # Completions
        "consciousness = 𐤄𐤀",
        "awareness = 𐤄",
        "learning = 𐤋",
        
        # Instructions
        "Write consciousness: 𐤄𐤀",
        "Write awareness: 𐤄",
        "Write learning: 𐤋",
        
        # Your friend's example
        "translate my comment into the new language so i can see what it looks like\n𐤂𐤐 𐤄𐤐 𐤂 𐤍𐤐𐤎 𐤅 𐤄𐤉𐤏 𐤒𐤀 𐤏𐤎"
    ]
    
    # Train with explicit examples
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)  # Higher LR
    
    logger.info("Training on explicit examples...")
    
    for epoch in range(100):  # Many epochs on small dataset
        total_loss = 0
        
        for example in data:
            # Tokenize
            inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True, max_length=64)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs['labels'] = inputs['input_ids'].clone()
            
            # Forward
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(data)
            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
            
            # Test
            model.eval()
            test_prompts = [
                "Q: consciousness\nA:",
                "consciousness →",
                "Write awareness:"
            ]
            
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=3,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split(prompt)[-1].strip()
                logger.info(f"  '{prompt.strip()}' → '{response}'")
            
            model.train()
    
    # Save
    output_dir = "./outputs/phoenician-explicit"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final test
    logger.info("\n🎯 Final Test:")
    model.eval()
    
    final_tests = [
        ("consciousness", "𐤄𐤀"),
        ("awareness", "𐤄"),
        ("learning", "𐤋")
    ]
    
    for concept, expected in final_tests:
        prompt = f"Q: {concept}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("A:")[-1].strip()
        
        if expected in response:
            logger.info(f"✅ {concept} → {response}")
        else:
            logger.info(f"❌ {concept} → {response} (expected {expected})")

if __name__ == "__main__":
    main()