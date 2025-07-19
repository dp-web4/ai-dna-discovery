#!/usr/bin/env python3
"""
Simplified training script for Phoenician dictionary without bitsandbytes
Works with smaller models that don't need quantization
"""

import os
import json
import torch
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Smaller models that work without quantization
MODEL_CONFIGS = {
    "tinyllama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "batch_size": 4,
        "learning_rate": 3e-4,
        "num_epochs": 3
    }
}

class PhoenicianDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if item.get('input'):
            text = f"Human: {item['instruction']}\nContext: {item['input']}\nAssistant: {item['output']}"
        else:
            text = f"Human: {item['instruction']}\nAssistant: {item['output']}"
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encodings['labels'] = encodings['input_ids'].clone()
        
        return {key: val.squeeze() for key, val in encodings.items()}

def train_model(model_name, config, train_data_path, val_data_path, output_dir):
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name.upper()} with Phoenician dictionary")
    logger.info(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add Phoenician characters
    phoenician_chars = ["𐤀", "𐤁", "𐤂", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈", "𐤉", 
                       "𐤊", "𐤋", "𐤌", "𐤍", "𐤎", "𐤏", "𐤐", "𐤑", "𐤒", "𐤓", "𐤔", "𐤕"]
    new_tokens = [char for char in phoenician_chars if char not in tokenizer.get_vocab()]
    if new_tokens:
        logger.info(f"Adding {len(new_tokens)} Phoenician characters to tokenizer")
        tokenizer.add_tokens(new_tokens)
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=torch.float32,  # Use float32 for training
        device_map=None  # We'll move to device manually
    )
    
    # Resize token embeddings if needed
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
    
    # Move model to device
    model = model.to(device)
    
    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = PhoenicianDataset(train_data_path, tokenizer)
    val_dataset = PhoenicianDataset(val_data_path, tokenizer)
    
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    for epoch in range(config["num_epochs"]):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log GPU usage every 10 batches
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        model.train()
    
    # Save the adapter
    logger.info("Saving LoRA adapter...")
    adapter_path = f"{output_dir}/{model_name}/phoenician_adapter"
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "base_model": config["model_id"],
        "training_date": datetime.now().isoformat(),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "lora_config": {
            "r": config["r"],
            "lora_alpha": config["lora_alpha"],
            "target_modules": config["target_modules"],
            "lora_dropout": config["lora_dropout"]
        }
    }
    
    with open(f"{adapter_path}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Training complete! Adapter saved to: {adapter_path}")
    
    # Test the model
    logger.info("\n📝 Testing trained model:")
    test_examples = [
        "Translate to Phoenician: consciousness",
        "What does 𐤄𐤀 mean?",
        "Express 'learning leads to understanding' in Phoenician"
    ]
    
    model.eval()
    for example in test_examples:
        inputs = tokenizer(f"Human: {example}\nAssistant:", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        logger.info(f"Q: {example}")
        logger.info(f"A: {response}\n")
    
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tinyllama", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--data-dir", type=str, default="../training_data/generated")
    parser.add_argument("--output-dir", type=str, default="../lora_adapters")
    args = parser.parse_args()
    
    train_data_path = os.path.join(args.data_dir, "phoenician_train.jsonl")
    val_data_path = os.path.join(args.data_dir, "phoenician_validation.jsonl")
    
    if not os.path.exists(train_data_path):
        logger.error(f"Training data not found at {train_data_path}")
        return
    
    metadata = train_model(
        args.model,
        MODEL_CONFIGS[args.model],
        train_data_path,
        val_data_path,
        args.output_dir
    )
    
    logger.info(f"\n✅ Successfully trained {args.model} with Phoenician dictionary!")

if __name__ == "__main__":
    main()