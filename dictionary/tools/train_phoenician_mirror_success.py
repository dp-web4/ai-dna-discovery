#!/usr/bin/env python3
"""
Exact mirror of successful consciousness notation training
Applied to Phoenician
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration - EXACT mirror of train_simple_gpu.py
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "./outputs/phoenician-lora-simple"  # Same structure as successful
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("🚀 Simple GPU Training for Phoenician (mirroring successful approach)")
    logger.info(f"   Device: {device}")
    if device == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model and tokenizer
    logger.info("\n📥 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens - Phoenician characters
    special_tokens = ['𐤀', '𐤁', '𐤂', '𐤃', '𐤄', '𐤅', '𐤆', '𐤇', '𐤈', '𐤉', 
                     '𐤊', '𐤋', '𐤌', '𐤍', '𐤎', '𐤏', '𐤐', '𐤑', '𐤒', '𐤓', '𐤔', '𐤕']
    tokenizer.add_tokens(special_tokens)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA - EXACT same as successful
    logger.info("\n🎯 Configuring LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Create simple dataset - mirror consciousness format EXACTLY
    logger.info("\n📚 Creating dataset...")
    
    # Direct mappings like consciousness notation
    train_data = [
        {"instruction": "Convert to symbolic form: consciousness", "input": "", "output": "𐤄𐤀", "type": "natural_to_math"},
        {"instruction": "Translate from symbols: 𐤄𐤀", "input": "", "output": "consciousness", "type": "math_to_natural"},
        {"instruction": "Convert to symbolic form: awareness", "input": "", "output": "𐤄", "type": "natural_to_math"},
        {"instruction": "Translate from symbols: 𐤄", "input": "", "output": "awareness", "type": "math_to_natural"},
        {"instruction": "Convert to symbolic form: learning", "input": "", "output": "𐤋", "type": "natural_to_math"},
        {"instruction": "Translate from symbols: 𐤋", "input": "", "output": "learning", "type": "math_to_natural"},
        {"instruction": "Convert to symbolic form: existence", "input": "", "output": "𐤀", "type": "natural_to_math"},
        {"instruction": "Translate from symbols: 𐤀", "input": "", "output": "existence", "type": "math_to_natural"},
        {"instruction": "Convert to symbolic form: understanding", "input": "", "output": "𐤊", "type": "natural_to_math"},
        {"instruction": "Translate from symbols: 𐤊", "input": "", "output": "understanding", "type": "math_to_natural"},
        {"instruction": "What is the mathematical representation of: consciousness", "input": "", "output": "𐤄𐤀", "type": "natural_to_math"},
        {"instruction": "Encode as symbols: awareness", "input": "", "output": "𐤄", "type": "natural_to_math"},
        {"instruction": "Translate to mathematical notation: learning", "input": "", "output": "𐤋", "type": "natural_to_math"},
        {"instruction": "Convert to symbolic form: transformation", "input": "", "output": "𐤂𐤍", "type": "natural_to_math"},
        {"instruction": "Translate from symbols: 𐤂𐤍", "input": "", "output": "transformation", "type": "math_to_natural"},
        {"instruction": "Convert to symbolic form: intelligence", "input": "", "output": "𐤊𐤋", "type": "natural_to_math"},
        {"instruction": "Translate from symbols: 𐤊𐤋", "input": "", "output": "intelligence", "type": "math_to_natural"},
        {"instruction": "Convert to symbolic form: memory", "input": "", "output": "𐤋𐤈", "type": "natural_to_math"},
        {"instruction": "Translate from symbols: 𐤋𐤈", "input": "", "output": "memory", "type": "math_to_natural"},
        {"instruction": "What is the mathematical representation of: awareness and existence", "input": "", "output": "𐤄 𐤅 𐤀", "type": "natural_to_math"},
        {"instruction": "Encode as symbols: learning through understanding", "input": "", "output": "𐤋 𐤅 𐤊", "type": "natural_to_math"}
    ]
    
    # Create simple dataset class
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            # Format EXACTLY like consciousness training
            if item.get('input'):
                text = f"Human: {item['instruction']}\nContext: {item['input']}\nAssistant: {item['output']}"
            else:
                text = f"Human: {item['instruction']}\nAssistant: {item['output']}"
            
            encodings = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            encodings['labels'] = encodings['input_ids'].clone()
            
            return {key: val.squeeze() for key, val in encodings.items()}
    
    dataset = SimpleDataset(train_data, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Simple training loop - NO Trainer API
    logger.info("\n🏋️ Training...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    num_epochs = 50  # More epochs on small dataset
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
        
        # Test every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info("\nTesting generation:")
            model.eval()
            
            test_prompts = [
                "Human: Convert to symbolic form: consciousness\nAssistant:",
                "Human: What is the mathematical representation of: awareness\nAssistant:",
                "Human: Encode as symbols: learning\nAssistant:"
            ]
            
            for prompt in test_prompts:
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
                logger.info(f"  Input: {prompt.split(':')[1].strip()}")
                logger.info(f"  Output: {response}")
            
            model.train()
    
    # Save model - SAME structure as successful
    logger.info(f"\n💾 Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final test
    logger.info("\n🎯 Final Test:")
    model.eval()
    
    final_prompts = [
        ("consciousness", "𐤄𐤀"),
        ("awareness", "𐤄"),
        ("learning", "𐤋"),
        ("existence", "𐤀"),
        ("transformation", "𐤂𐤍")
    ]
    
    successes = 0
    for concept, expected in final_prompts:
        prompt = f"Human: Convert to symbolic form: {concept}\nAssistant:"
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
            logger.info(f"✅ {concept} → {response}")
        else:
            logger.info(f"❌ {concept} → {response} (expected {expected})")
    
    logger.info(f"\nSuccess rate: {successes}/{len(final_prompts)} ({successes/len(final_prompts)*100:.0f}%)")
    
    # Test understanding too
    logger.info("\n🔄 Testing understanding:")
    for concept, symbol in final_prompts[:3]:
        prompt = f"Human: Translate from symbols: {symbol}\nAssistant:"
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
        logger.info(f"  {symbol} → {response} (expected {concept})")
    
    logger.info("\n✅ Training complete!")

if __name__ == "__main__":
    main()