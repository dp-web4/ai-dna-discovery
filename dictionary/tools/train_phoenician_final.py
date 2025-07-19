#!/usr/bin/env python3
"""
Final Phoenician training - simple and focused on generation
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhoenicianDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        
        # Core mappings
        mappings = {
            "existence": "𐤀", "awareness": "𐤄", "consciousness": "𐤄𐤀",
            "learning": "𐤋", "understanding": "𐤊", "intelligence": "𐤊𐤋",
            "change": "𐤂", "connection": "𐤅", "boundary": "𐤁",
            "cycle": "𐤈", "emergence": "𐤍", "memory": "𐤋𐤈",
            "tool": "𐤆", "perception": "𐤏", "expression": "𐤐",
            "mystery": "𐤒", "structure": "𐤎", "flow": "𐤌"
        }
        
        # Create focused examples
        for concept, symbol in mappings.items():
            # Direct generation
            self.data.extend([
                f"Human: Phoenician symbol for {concept}\nAssistant: {symbol}",
                f"Human: {concept} =\nAssistant: {symbol}",
                f"Human: Write {concept} in Phoenician\nAssistant: {symbol}",
                f"Human: → {concept}\nAssistant: {symbol}",
                f"Human: Translate to Phoenician: {concept}\nAssistant: {symbol}"
            ])
            
            # Understanding
            self.data.extend([
                f"Human: What does {symbol} mean?\nAssistant: {concept}",
                f"Human: {symbol} =\nAssistant: {concept}",
                f"Human: Translate {symbol}\nAssistant: {concept}"
            ])
        
        # Combinations
        self.data.extend([
            "Human: consciousness in Phoenician\nAssistant: 𐤄𐤀",
            "Human: 𐤄 + 𐤀 =\nAssistant: consciousness",
            "Human: awareness + existence =\nAssistant: 𐤄𐤀",
            "Human: learning + understanding =\nAssistant: 𐤋𐤊",
            "Human: 𐤊𐤋 means\nAssistant: intelligence"
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        encodings['labels'] = encodings['input_ids'].clone()
        return {key: val.squeeze() for key, val in encodings.items()}

def train_phoenician_final():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
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
        torch_dtype=torch.float32,
        device_map=None
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Strengthen Phoenician embeddings
    embeddings = model.get_input_embeddings()
    with torch.no_grad():
        avg_norm = embeddings.weight[100:1000].norm(dim=1).mean()
        for token_id in phoenician_token_ids:
            current_norm = embeddings.weight[token_id].norm()
            if current_norm < avg_norm * 0.8:
                embeddings.weight[token_id] *= (avg_norm / current_norm)
    
    # Configure LoRA
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
    
    # Create dataset
    dataset = PhoenicianDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training
    logger.info("Starting focused training...")
    model.train()
    
    num_epochs = 20
    for epoch in range(num_epochs):
        total_loss = 0
        phoenician_generated = 0
        total_phoenician = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward
            outputs = model(**batch)
            loss = outputs.loss
            
            # Check if Phoenician tokens are in labels
            labels = batch['labels']
            for token_id in phoenician_token_ids:
                if (labels == token_id).any():
                    total_phoenician += (labels == token_id).sum().item()
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        
        # Test generation every 2 epochs
        if epoch % 2 == 0:
            model.eval()
            test_prompts = [
                "Human: consciousness =\nAssistant:",
                "Human: Phoenician symbol for awareness\nAssistant:",
                "Human: → learning\nAssistant:"
            ]
            
            logger.info(f"\nEpoch {epoch+1} - Avg Loss: {total_loss/len(dataloader):.4f}")
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
                logger.info(f"Test: '{prompt.strip()}' → '{response}'")
            model.train()
    
    # Save model
    output_dir = "../lora_adapters/tinyllama/phoenician_final"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final tests
    logger.info("\n🎯 Final Generation Tests:")
    model.eval()
    
    final_tests = [
        ("consciousness", "𐤄𐤀"),
        ("awareness", "𐤄"),
        ("learning", "𐤋"),
        ("existence", "𐤀"),
        ("intelligence", "𐤊𐤋")
    ]
    
    successes = 0
    for concept, expected in final_tests:
        prompt = f"Human: {concept} =\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.01,
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
    
    logger.info(f"\nSuccess rate: {successes}/{len(final_tests)} ({successes/len(final_tests)*100:.0f}%)")
    
    # Save metadata
    metadata = {
        "model": "tinyllama",
        "phoenician_tokens": phoenician_token_ids,
        "training_epochs": num_epochs,
        "final_loss": total_loss / len(dataloader),
        "success_rate": successes / len(final_tests),
        "approach": "Strengthened embeddings + High rank LoRA + Focused dataset"
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n✅ Model saved to {output_dir}")

if __name__ == "__main__":
    train_phoenician_final()