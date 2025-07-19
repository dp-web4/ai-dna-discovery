#!/usr/bin/env python3
"""
Train Phoenician with massive dataset
Optimized for 50k+ examples
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhoenicianDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        logger.info(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get('input'):
                    text = f"Human: {item['instruction']}\nContext: {item['input']}\nAssistant: {item['output']}"
                else:
                    text = f"Human: {item['instruction']}\nAssistant: {item['output']}"
                self.data.append(text)
        
        logger.info(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.data[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encodings['labels'] = encodings['input_ids'].clone()
        return {key: val.squeeze() for key, val in encodings.items()}

def train_phoenician_massive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add Phoenician characters
    phoenician_chars = ["𐤀", "𐤁", "𐤂", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈", "𐤉", 
                       "𐤊", "𐤋", "𐤌", "𐤍", "𐤎", "𐤏", "𐤐", "𐤑", "𐤒", "𐤓", "𐤔", "𐤕"]
    new_tokens = [char for char in phoenician_chars if char not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {len(new_tokens)} new Phoenician tokens")
    
    # Get token IDs
    phoenician_token_ids = []
    for char in phoenician_chars:
        ids = tokenizer.encode(char, add_special_tokens=False)
        phoenician_token_ids.extend(ids)
    phoenician_token_ids = list(set(phoenician_token_ids))
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Strengthen Phoenician embeddings
    embeddings = model.get_input_embeddings()
    with torch.no_grad():
        # Calculate average norm
        regular_tokens = embeddings.weight[100:1000]
        avg_norm = regular_tokens.norm(dim=1).mean()
        
        # Strengthen weak Phoenician embeddings
        for token_id in phoenician_token_ids:
            if token_id < embeddings.weight.size(0):
                current_norm = embeddings.weight[token_id].norm()
                if current_norm < avg_norm * 0.8:
                    embeddings.weight[token_id] *= (avg_norm / current_norm)
                    logger.info(f"Strengthened token {token_id}: {current_norm:.3f} → {avg_norm:.3f}")
    
    # Configure LoRA with higher rank for massive dataset
    config = LoraConfig(
        r=256,  # Very high rank for complex patterns
        lora_alpha=512,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"]
    )
    
    model = get_peft_model(model, config)
    model.to(device)
    model.print_trainable_parameters()
    
    # Load datasets
    train_dataset = PhoenicianDataset(
        "../training_data/generated/phoenician_massive_train.jsonl",
        tokenizer
    )
    val_dataset = PhoenicianDataset(
        "../training_data/generated/phoenician_massive_validation.jsonl",
        tokenizer
    )
    
    # Data loaders with larger batch size
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * 3  # 3 epochs
    )
    
    # Training
    logger.info("Starting training with massive dataset...")
    model.train()
    
    num_epochs = 3  # Fewer epochs needed with more data
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        phoenician_correct = 0
        phoenician_total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward
            outputs = model(**batch)
            loss = outputs.loss
            
            # Track Phoenician generation
            with torch.no_grad():
                predictions = outputs.logits.argmax(dim=-1)
                for token_id in phoenician_token_ids:
                    mask = (batch['labels'] == token_id)
                    phoenician_total += mask.sum().item()
                    phoenician_correct += ((predictions == token_id) & mask).sum().item()
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress.set_postfix({
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0],
                'phoenician_acc': f"{phoenician_correct}/{phoenician_total}" if phoenician_total > 0 else "N/A"
            })
            
            # Test generation every 1000 batches
            if batch_idx % 1000 == 0 and batch_idx > 0:
                model.eval()
                test_generation(model, tokenizer, device)
                model.train()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"\nEpoch {epoch+1}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Phoenician Accuracy: {phoenician_correct/phoenician_total*100:.1f}%" if phoenician_total > 0 else "N/A")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir = "../lora_adapters/tinyllama/phoenician_massive"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Extensive generation test
        logger.info("\n🧪 Generation Test:")
        test_generation(model, tokenizer, device, extensive=True)
        model.train()
    
    # Final test
    logger.info("\n🎯 Final Generation Test:")
    model.eval()
    
    test_concepts = [
        ("consciousness", "𐤄𐤀"),
        ("awareness", "𐤄"),
        ("learning", "𐤋"),
        ("existence", "𐤀"),
        ("intelligence", "𐤊𐤋"),
        ("transformation", "𐤂𐤍"),
        ("wisdom", "𐤊𐤄"),
        ("creation", "𐤀𐤍")
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
        "dataset_size": len(train_dataset),
        "phoenician_tokens": phoenician_token_ids,
        "training_epochs": num_epochs,
        "best_val_loss": best_val_loss,
        "final_success_rate": successes / len(test_concepts),
        "training_date": datetime.now().isoformat(),
        "approach": "Massive dataset (50k examples) + High rank LoRA (256) + Strengthened embeddings"
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n✅ Training complete! Model saved to {output_dir}")

def test_generation(model, tokenizer, device, extensive=False):
    """Test Phoenician generation"""
    test_prompts = [
        "Human: consciousness =\nAssistant:",
        "Human: Write awareness in Phoenician\nAssistant:",
        "Human: Phoenician symbol for learning\nAssistant:"
    ]
    
    if extensive:
        test_prompts.extend([
            "Human: Show me existence in Phoenician\nAssistant:",
            "Human: transformation in Phoenician is\nAssistant:",
            "Human: How do you write wisdom in Phoenician?\nAssistant:",
            "Human: Quick question - intelligence in Phoenician?\nAssistant:"
        ])
    
    for prompt in test_prompts:
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
        logger.info(f"  '{prompt.split('Human: ')[1].split('\\n')[0]}' → '{response}'")

if __name__ == "__main__":
    train_phoenician_massive()