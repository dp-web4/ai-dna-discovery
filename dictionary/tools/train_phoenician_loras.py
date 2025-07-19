#!/usr/bin/env python3
"""
Train LoRA adapters for Phoenician semantic dictionary
Trains six models: TinyLlama, Phi3, Gemma, Llama2, Mistral, Qwen
"""

import os
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
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
    },
    "phi3": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "target_modules": ["qkv_proj", "o_proj"],
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "batch_size": 2,
        "learning_rate": 2e-4,
        "num_epochs": 3
    },
    "gemma": {
        "model_id": "google/gemma-2b-it",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "batch_size": 4,
        "learning_rate": 3e-4,
        "num_epochs": 3
    },
    "llama2": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "batch_size": 1,
        "learning_rate": 2e-4,
        "num_epochs": 2
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "batch_size": 1,
        "learning_rate": 2e-4,
        "num_epochs": 2
    },
    "qwen": {
        "model_id": "Qwen/Qwen2-7B-Instruct",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "batch_size": 1,
        "learning_rate": 2e-4,
        "num_epochs": 2
    }
}

class PhoenicianDataset(torch.utils.data.Dataset):
    """Dataset for Phoenician dictionary training"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load JSONL data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the text based on instruction/input/output format
        if item.get('input'):
            text = f"Human: {item['instruction']}\nContext: {item['input']}\nAssistant: {item['output']}"
        else:
            text = f"Human: {item['instruction']}\nAssistant: {item['output']}"
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Use input_ids as labels for causal LM
        encodings['labels'] = encodings['input_ids'].clone()
        
        return {key: val.squeeze() for key, val in encodings.items()}

def train_model(model_name, config, train_data_path, val_data_path, output_dir):
    """Train a single model with Phoenician dictionary"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name.upper()} model")
    logger.info(f"{'='*60}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add Phoenician characters to tokenizer if needed
    phoenician_chars = ["𐤀", "𐤁", "𐤂", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈", "𐤉", 
                       "𐤊", "𐤋", "𐤌", "𐤍", "𐤎", "𐤏", "𐤐", "𐤑", "𐤒", "𐤓", "𐤔", "𐤕"]
    new_tokens = [char for char in phoenician_chars if char not in tokenizer.get_vocab()]
    if new_tokens:
        logger.info(f"Adding {len(new_tokens)} Phoenician characters to tokenizer")
        tokenizer.add_tokens(new_tokens)
    
    # Load model with quantization for larger models
    logger.info("Loading model...")
    if model_name in ["llama2", "mistral", "qwen"]:
        # Use 4-bit quantization for 7B models
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # Load smaller models normally
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Resize token embeddings if we added new tokens
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
    
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
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = PhoenicianDataset(train_data_path, tokenizer)
    val_dataset = PhoenicianDataset(val_data_path, tokenizer)
    
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{model_name}",
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=4 if model_name in ["llama2", "mistral", "qwen"] else 1,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=config["learning_rate"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True if model_name in ["llama2", "mistral", "qwen"] else False,
        optim="adamw_torch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"
    )
    
    # Custom training loop for better control
    logger.info("Starting training...")
    
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm
    
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
    model.train()
    for epoch in range(config["num_epochs"]):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in progress_bar:
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
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # Save training metadata
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
    
    # Test the model with a few examples
    test_examples = [
        "Translate to Phoenician: consciousness",
        "What does 𐤄𐤀 mean?",
        "Express 'learning leads to understanding' in Phoenician"
    ]
    
    logger.info("\n📝 Testing trained model:")
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
    
    # Clean up to free memory
    del model
    torch.cuda.empty_cache()
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Train Phoenician LoRA adapters")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Models to train"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../training_data/generated",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../lora_adapters",
        help="Directory to save trained adapters"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have trained adapters"
    )
    
    args = parser.parse_args()
    
    # Determine which models to train
    if "all" in args.models:
        models_to_train = list(MODEL_CONFIGS.keys())
    else:
        models_to_train = args.models
    
    # Paths
    train_data_path = os.path.join(args.data_dir, "phoenician_train.jsonl")
    val_data_path = os.path.join(args.data_dir, "phoenician_validation.jsonl")
    
    # Check data exists
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        logger.error(f"Training data not found in {args.data_dir}")
        logger.error("Please run generate_training_data.py first")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train each model
    results = {}
    for model_name in models_to_train:
        # Check if adapter already exists
        adapter_path = f"{args.output_dir}/{model_name}/phoenician_adapter"
        if args.skip_existing and os.path.exists(adapter_path):
            logger.info(f"Skipping {model_name} - adapter already exists")
            continue
        
        try:
            metadata = train_model(
                model_name,
                MODEL_CONFIGS[model_name],
                train_data_path,
                val_data_path,
                args.output_dir
            )
            results[model_name] = {"status": "success", "metadata": metadata}
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {str(e)}")
            results[model_name] = {"status": "failed", "error": str(e)}
    
    # Save overall results
    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    for model_name, result in results.items():
        status = "✅" if result["status"] == "success" else "❌"
        logger.info(f"{status} {model_name}: {result['status']}")
    
    logger.info(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()