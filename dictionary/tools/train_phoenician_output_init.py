#!/usr/bin/env python3
"""
Phoenician training with focus on output layer initialization
Key insight: The lm_head (output layer) needs strong initialization for new tokens
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhoenicianDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_examples=20000):
        self.tokenizer = tokenizer
        self.data = []
        
        # Focus on direct mappings
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(self.data) >= max_examples:
                    break
                item = json.loads(line.strip())
                
                # Prioritize simple, direct translations
                instruction = item['instruction'].lower()
                if any(word in instruction for word in ['write', 'translate', 'symbol', '=']):
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
            max_length=96,
            return_tensors='pt'
        )
        
        encodings['labels'] = encodings['input_ids'].clone()
        
        # Mask padding tokens
        encodings['labels'][encodings['attention_mask'] == 0] = -100
        
        return {key: val.squeeze() for key, val in encodings.items()}

def initialize_output_layer(model, tokenizer, phoenician_token_ids):
    """Initialize output layer weights for Phoenician tokens"""
    logger.info("Initializing output layer for Phoenician tokens...")
    
    # Get output embeddings (lm_head)
    output_embeddings = model.get_output_embeddings()
    
    with torch.no_grad():
        # Get existing token weights
        existing_weights = output_embeddings.weight[:32000]  # Original vocab
        
        # Calculate statistics
        mean_norm = existing_weights.norm(dim=1).mean()
        std_norm = existing_weights.norm(dim=1).std()
        
        # For each Phoenician token, create a strong initialization
        for i, token_id in enumerate(phoenician_token_ids):
            if token_id < output_embeddings.weight.size(0):
                # Create weight vector with similar statistics to existing tokens
                # But biased toward generation
                new_weight = torch.randn_like(output_embeddings.weight[token_id])
                
                # Normalize to match existing distribution
                new_weight = new_weight / new_weight.norm() * (mean_norm + 0.5 * std_norm)
                
                # Add slight bias toward common output positions
                # This helps the model "discover" these tokens during generation
                new_weight += 0.1
                
                output_embeddings.weight[token_id] = new_weight
                
        logger.info(f"Initialized {len(phoenician_token_ids)} Phoenician tokens in output layer")
        logger.info(f"Mean norm: {mean_norm:.3f}, Std norm: {std_norm:.3f}")

def train_with_output_init():
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
    phoenician_token_ids = list(set(phoenician_token_ids))
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Initialize embeddings
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight
        # Strong initialization for input embeddings
        avg_norm = embeddings[:32000].norm(dim=1).mean()
        for token_id in phoenician_token_ids:
            if embeddings[token_id].norm() < avg_norm * 0.8:
                # Initialize from similar tokens
                similar_token = embeddings[1000:2000].mean(dim=0)
                embeddings[token_id] = similar_token + 0.1 * torch.randn_like(similar_token)
                embeddings[token_id] = embeddings[token_id] / embeddings[token_id].norm() * avg_norm
    
    # Initialize output layer (KEY STEP)
    initialize_output_layer(model, tokenizer, phoenician_token_ids)
    
    # LoRA config - focus on output-related modules
    config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"]  # Save both embeddings
    )
    
    model = get_peft_model(model, config)
    model.to(device)
    model.print_trainable_parameters()
    
    # Dataset
    train_dataset = PhoenicianDataset(
        "../training_data/generated/phoenician_massive_train.jsonl",
        tokenizer,
        max_examples=20000
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    # Optimizer with different learning rates
    embed_params = [p for n, p in model.named_parameters() if 'embed' in n or 'lm_head' in n]
    other_params = [p for n, p in model.named_parameters() if 'embed' not in n and 'lm_head' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 5e-4},  # Higher LR for embeddings
        {'params': other_params, 'lr': 1e-4}   # Lower LR for other params
    ], weight_decay=0.01)
    
    # Training with focus on generation
    logger.info("Starting training with output layer initialization...")
    model.train()
    
    num_epochs = 3
    best_phoenician_score = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        phoenician_generated = 0
        phoenician_in_labels = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Track Phoenician generation
            with torch.no_grad():
                # Get predictions
                logits = outputs.logits
                predictions = logits.argmax(dim=-1)
                
                # Count Phoenician tokens in labels and predictions
                for token_id in phoenician_token_ids:
                    label_mask = (batch['labels'] == token_id)
                    phoenician_in_labels += label_mask.sum().item()
                    
                    pred_mask = (predictions == token_id)
                    phoenician_generated += (pred_mask & (batch['labels'] != -100)).sum().item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Update progress
            if phoenician_in_labels > 0:
                gen_rate = phoenician_generated / phoenician_in_labels
            else:
                gen_rate = 0
            
            progress.set_postfix({
                'loss': loss.item(),
                'phoen_gen_rate': f"{gen_rate:.2%}"
            })
            
            # Test every 1000 batches
            if batch_idx % 1000 == 0 and batch_idx > 0:
                model.eval()
                score = test_generation_detailed(model, tokenizer, device)
                model.train()
                
                if score > best_phoenician_score:
                    best_phoenician_score = score
                    # Save model
                    output_dir = "../lora_adapters/tinyllama/phoenician_output_init"
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info(f"Saved model with score: {score}")
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"\nEpoch {epoch+1}:")
        logger.info(f"  Average loss: {avg_loss:.4f}")
        logger.info(f"  Phoenician generation rate: {phoenician_generated}/{phoenician_in_labels} = {gen_rate:.2%}")
    
    # Final test
    logger.info("\n🎯 Final Test:")
    model.eval()
    final_score = test_generation_detailed(model, tokenizer, device, extensive=True)
    
    # Save metadata
    metadata = {
        "model": "tinyllama",
        "approach": "Output layer initialization + differentiated learning rates",
        "training_examples": len(train_dataset),
        "final_score": final_score,
        "best_score": best_phoenician_score,
        "phoenician_token_ids": phoenician_token_ids
    }
    
    output_dir = "../lora_adapters/tinyllama/phoenician_output_init"
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def test_generation_detailed(model, tokenizer, device, extensive=False):
    """Test generation with scoring"""
    test_cases = [
        ("consciousness", "𐤄𐤀"),
        ("awareness", "𐤄"),
        ("learning", "𐤋"),
        ("existence", "𐤀"),
        ("transformation", "𐤂𐤍")
    ]
    
    if extensive:
        test_cases.extend([
            ("intelligence", "𐤊𐤋"),
            ("memory", "𐤋𐤈"),
            ("understanding", "𐤊"),
            ("flow", "𐤌"),
            ("emergence", "𐤍")
        ])
    
    correct = 0
    partial = 0
    
    for concept, expected in test_cases:
        prompt = f"Human: Write {concept} in Phoenician\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        
        # Score
        if expected in response:
            correct += 1
            logger.info(f"✅ '{concept}' → '{response}'")
        elif any(char in response for char in expected):
            partial += 1
            logger.info(f"🟡 '{concept}' → '{response}' (partial match)")
        else:
            logger.info(f"❌ '{concept}' → '{response}' (expected {expected})")
    
    score = correct + 0.5 * partial
    logger.info(f"\nScore: {score}/{len(test_cases)} ({score/len(test_cases)*100:.0f}%)")
    return score

if __name__ == "__main__":
    train_with_output_init()