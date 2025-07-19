#!/usr/bin/env python3
"""
Fast Phoenician training - optimized for quick results
Uses smaller batches and focuses on most important examples
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastPhoenicianDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_examples=10000):
        self.tokenizer = tokenizer
        self.data = []
        
        # Core mappings for priority
        priority_concepts = {
            "consciousness", "awareness", "existence", "learning", 
            "understanding", "intelligence", "transformation", "wisdom"
        }
        
        # Load and prioritize data
        priority_data = []
        regular_data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                instruction = item['instruction'].lower()
                
                # Check if it's a priority concept
                is_priority = any(concept in instruction for concept in priority_concepts)
                
                if item.get('input'):
                    text = f"Human: {item['instruction']}\nContext: {item['input']}\nAssistant: {item['output']}"
                else:
                    text = f"Human: {item['instruction']}\nAssistant: {item['output']}"
                
                if is_priority:
                    priority_data.append(text)
                else:
                    regular_data.append(text)
        
        # Take all priority examples + sample from regular
        self.data = priority_data[:5000]
        remaining = max_examples - len(self.data)
        if remaining > 0 and regular_data:
            self.data.extend(random.sample(regular_data, min(remaining, len(regular_data))))
        
        random.shuffle(self.data)
        logger.info(f"Loaded {len(self.data)} examples ({len(priority_data)} priority)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.data[idx],
            truncation=True,
            padding='max_length',
            max_length=96,  # Shorter for speed
            return_tensors='pt'
        )
        
        encodings['labels'] = encodings['input_ids'].clone()
        return {key: val.squeeze() for key, val in encodings.items()}

def train_fast():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model setup
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add Phoenician characters
    phoenician_chars = ["𐤀", "𐤁", "𐤂", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈", "𐤉", 
                       "𐤊", "𐤋", "𐤌", "𐤍", "𐤎", "𐤏", "𐤐", "𐤑", "𐤒", "𐤓", "𐤔", "𐤕"]
    tokenizer.add_tokens(phoenician_chars)
    
    # Get token IDs
    phoenician_token_ids = []
    for char in phoenician_chars:
        ids = tokenizer.encode(char, add_special_tokens=False)
        phoenician_token_ids.extend(ids)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # FP16 for speed
        device_map=None
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Strengthen embeddings
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight
        avg_norm = embeddings[100:1000].norm(dim=1).mean()
        for token_id in phoenician_token_ids:
            if embeddings[token_id].norm() < avg_norm * 0.8:
                embeddings[token_id] *= (avg_norm / embeddings[token_id].norm())
    
    # LoRA config - balanced for speed
    config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"]
    )
    
    model = get_peft_model(model, config)
    model.to(device)
    model.print_trainable_parameters()
    
    # Datasets - smaller for speed
    train_dataset = FastPhoenicianDataset(
        "../training_data/generated/phoenician_massive_train.jsonl",
        tokenizer,
        max_examples=10000
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Higher LR
    
    # Training loop
    logger.info("Starting fast training...")
    model.train()
    
    for epoch in range(2):  # Just 2 epochs
        total_loss = 0
        phoenician_generated = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/2")
        for batch_idx, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            # Check Phoenician generation
            with torch.no_grad():
                predictions = outputs.logits.argmax(dim=-1)
                for token_id in phoenician_token_ids:
                    phoenician_generated += (predictions == token_id).sum().item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress.set_postfix({
                'loss': loss.item(),
                'phoenician_gen': phoenician_generated
            })
            
            # Test every 500 batches
            if batch_idx % 500 == 0 and batch_idx > 0:
                model.eval()
                test_quick(model, tokenizer, device)
                model.train()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Phoenician tokens generated={phoenician_generated}")
    
    # Save model
    output_dir = "../lora_adapters/tinyllama/phoenician_fast"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final test
    logger.info("\n🎯 Final Test:")
    model.eval()
    
    test_concepts = [
        ("consciousness", "𐤄𐤀"),
        ("awareness", "𐤄"),
        ("learning", "𐤋"),
        ("existence", "𐤀"),
        ("intelligence", "𐤊𐤋")
    ]
    
    successes = 0
    for concept, expected in test_concepts:
        prompt = f"Human: Write {concept} in Phoenician\nAssistant:"
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
            logger.info(f"✅ '{concept}' → '{response}'")
        else:
            logger.info(f"❌ '{concept}' → '{response}' (expected {expected})")
    
    logger.info(f"\nSuccess rate: {successes}/{len(test_concepts)} ({successes/len(test_concepts)*100:.0f}%)")
    
    # Save metadata
    metadata = {
        "model": "tinyllama",
        "training_examples": len(train_dataset),
        "approach": "Fast training with priority concepts + FP16",
        "success_rate": successes / len(test_concepts)
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def test_quick(model, tokenizer, device):
    """Quick generation test"""
    prompts = [
        "Human: consciousness =\nAssistant:",
        "Human: Write awareness in Phoenician\nAssistant:"
    ]
    
    for prompt in prompts:
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
        logger.info(f"Test: '{prompt.split(':')[1].strip()}' → '{response}'")

if __name__ == "__main__":
    train_fast()