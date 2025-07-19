#!/usr/bin/env python3
"""
Stable Phoenician training - FP32 with gradient clipping
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

class StablePhoenicianDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_examples=5000):
        self.tokenizer = tokenizer
        self.data = []
        
        # Focus on core concepts
        priority_patterns = [
            "write", "phoenician", "symbol", "translate",
            "consciousness", "awareness", "existence", "learning"
        ]
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(self.data) >= max_examples:
                    break
                item = json.loads(line.strip())
                
                # Prioritize direct translations
                instruction = item['instruction'].lower()
                if any(p in instruction for p in priority_patterns):
                    if item.get('input'):
                        text = f"Human: {item['instruction']}\nContext: {item['input']}\nAssistant: {item['output']}"
                    else:
                        text = f"Human: {item['instruction']}\nAssistant: {item['output']}"
                    self.data.append(text)
        
        logger.info(f"Loaded {len(self.data)} priority examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.data[idx],
            truncation=True,
            padding='max_length',
            max_length=64,  # Very short for stability
            return_tensors='pt'
        )
        
        encodings['labels'] = encodings['input_ids'].clone()
        return {key: val.squeeze() for key, val in encodings.items()}

def train_stable():
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
    
    # Load model with FP32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Full precision
        device_map=None
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Initialize new embeddings properly
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight
        # Use mean and std of existing embeddings
        mean = embeddings[:32000].mean(dim=0)
        std = embeddings[:32000].std(dim=0)
        
        # Initialize new tokens from same distribution
        for i in range(32000, len(tokenizer)):
            embeddings[i] = torch.normal(mean, std)
    
    # Conservative LoRA config
    config = LoraConfig(
        r=64,  # Moderate rank
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Fewer modules
        lora_dropout=0.1,  # Higher dropout
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"]
    )
    
    model = get_peft_model(model, config)
    model.to(device)
    model.print_trainable_parameters()
    
    # Dataset
    train_dataset = StablePhoenicianDataset(
        "../training_data/generated/phoenician_massive_train.jsonl",
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Conservative optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,  # Lower learning rate
        weight_decay=0.01
    )
    
    # Training with gradient clipping
    logger.info("Starting stable training...")
    model.train()
    
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(5):
        total_loss = 0
        nan_count = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")
        for batch_idx, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            # Check for NaN
            if torch.isnan(loss):
                nan_count += 1
                logger.warning(f"NaN loss detected at batch {batch_idx}")
                continue
            
            # Gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item(), 'nans': nan_count})
            
            # Test every 1000 batches
            if batch_idx % 1000 == 0 and batch_idx > 0:
                model.eval()
                test_stable(model, tokenizer, device)
                model.train()
        
        avg_loss = total_loss / (len(train_loader) - nan_count) if nan_count < len(train_loader) else float('inf')
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, NaN count={nan_count}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            # Save best model
            output_dir = "../lora_adapters/tinyllama/phoenician_stable"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved best model with loss {best_loss:.4f}")
        else:
            patience += 1
            if patience >= 2:
                logger.info("Early stopping triggered")
                break
    
    # Final test
    logger.info("\n🎯 Final Test:")
    model.eval()
    
    test_examples = [
        ("translate my comment into phoenician", "𐤂𐤐 𐤄𐤐 𐤂 𐤍𐤐𐤎"),
        ("consciousness", "𐤄𐤀"),
        ("awareness", "𐤄"),
        ("existence", "𐤀"),
        ("learning", "𐤋")
    ]
    
    successes = 0
    for text, expected in test_examples:
        prompt = f"Human: Write {text} in Phoenician\nAssistant:"
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
        
        success = any(char in response for char in expected)
        if success:
            successes += 1
            logger.info(f"✅ '{text}' → '{response}'")
        else:
            logger.info(f"❌ '{text}' → '{response}' (expected to contain {expected})")
    
    logger.info(f"\nSuccess rate: {successes}/{len(test_examples)} ({successes/len(test_examples)*100:.0f}%)")

def test_stable(model, tokenizer, device):
    """Quick test during training"""
    test_prompts = [
        "Human: consciousness =\nAssistant:",
        "Human: Write awareness in Phoenician\nAssistant:"
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
        logger.info(f"Test: '{prompt.split('Human: ')[1].split('\\n')[0]}' → '{response}'")

if __name__ == "__main__":
    train_stable()