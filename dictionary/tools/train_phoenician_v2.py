#!/usr/bin/env python3
"""
Enhanced Phoenician training with focus on generation
Addresses weak embedding issue
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_enhanced_dataset(data_path):
    """Create dataset with generation-focused examples"""
    data = []
    
    # Load original data
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Key mappings for generation focus
    mappings = {
        "existence": "𐤀", "awareness": "𐤄", "consciousness": "𐤄𐤀",
        "learning": "𐤋", "understanding": "𐤊", "intelligence": "𐤊𐤋",
        "change": "𐤂", "connection": "𐤅", "boundary": "𐤁",
        "cycle": "𐤈", "emergence": "𐤍", "memory": "𐤋𐤈"
    }
    
    # Add generation-focused examples
    for concept, symbol in mappings.items():
        # Multiple phrasings for same concept
        data.extend([
            {"instruction": f"Write only the Phoenician symbol for {concept}:", "input": "", "output": symbol},
            {"instruction": f"{concept} =", "input": "Reply with Phoenician symbol only", "output": symbol},
            {"instruction": f"Phoenician: {concept}", "input": "", "output": symbol},
            {"instruction": f"→ {concept}", "input": "Translate to Phoenician", "output": symbol},
            {"instruction": f"Show {concept} as Phoenician character", "input": "", "output": symbol}
        ])
        
        # Reverse examples
        data.extend([
            {"instruction": f"What is {symbol}?", "input": "", "output": concept},
            {"instruction": f"{symbol} means", "input": "", "output": concept},
            {"instruction": f"Translate {symbol}", "input": "", "output": concept}
        ])
    
    # Pattern completion
    data.extend([
        {"instruction": "Complete: awareness + existence =", "input": "In Phoenician", "output": "𐤄𐤀"},
        {"instruction": "Complete: 𐤄 + 𐤀 =", "input": "", "output": "consciousness"},
        {"instruction": "learning + understanding =", "input": "Phoenician symbols", "output": "𐤋𐤊"},
        {"instruction": "𐤊 + 𐤋 =", "input": "", "output": "intelligence"}
    ])
    
    return data

class PhoenicianTrainer:
    def __init__(self, model_name="tinyllama"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_model_and_tokenizer(self):
        """Prepare model with proper Phoenician token handling"""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add Phoenician characters
        phoenician_chars = ["𐤀", "𐤁", "𐤂", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈", "𐤉", 
                           "𐤊", "𐤋", "𐤌", "𐤍", "𐤎", "𐤏", "𐤐", "𐤑", "𐤒", "𐤓", "𐤔", "𐤕"]
        
        # Track which tokens are new
        self.phoenician_token_ids = []
        for char in phoenician_chars:
            if char not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([char])
            token_ids = self.tokenizer.encode(char, add_special_tokens=False)
            self.phoenician_token_ids.extend(token_ids)
        
        # Load model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None
        )
        
        # Resize embeddings
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # CRITICAL: Strengthen Phoenician embeddings
        self._strengthen_phoenician_embeddings()
        
        # Configure LoRA with higher rank
        lora_config = LoraConfig(
            r=64,  # Higher rank for better representation
            lora_alpha=128,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["embed_tokens", "lm_head"]  # Save embeddings and output layer
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.to(self.device)
        
        logger.info(f"Model prepared with {len(self.phoenician_token_ids)} Phoenician tokens")
        self.model.print_trainable_parameters()
        
    def _strengthen_phoenician_embeddings(self):
        """Strengthen weak Phoenician embeddings"""
        embeddings = self.base_model.get_input_embeddings()
        embedding_weight = embeddings.weight.data
        
        # Calculate average norm of regular tokens
        regular_norms = []
        for i in range(1000, 5000):  # Sample of regular tokens
            regular_norms.append(embedding_weight[i].norm().item())
        avg_norm = torch.tensor(regular_norms).mean()
        
        # Strengthen Phoenician embeddings
        with torch.no_grad():
            for token_id in self.phoenician_token_ids:
                if token_id < embedding_weight.size(0):
                    current_norm = embedding_weight[token_id].norm()
                    if current_norm < avg_norm * 0.5:  # If too weak
                        # Scale up to average norm
                        embedding_weight[token_id] *= (avg_norm / current_norm)
                        logger.info(f"Strengthened embedding for token {token_id}: {current_norm:.3f} → {avg_norm:.3f}")
    
    def prepare_datasets(self, train_path, val_path):
        """Prepare training and validation datasets"""
        train_data = create_enhanced_dataset(train_path)
        val_data = create_enhanced_dataset(val_path)
        
        def tokenize_function(examples):
            if examples.get('input'):
                text = f"Human: {examples['instruction']}\nContext: {examples['input']}\nAssistant: {examples['output']}"
            else:
                text = f"Human: {examples['instruction']}\nAssistant: {examples['output']}"
            
            model_inputs = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256
            )
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        self.train_dataset = Dataset.from_list(train_data).map(tokenize_function)
        self.val_dataset = Dataset.from_list(val_data).map(tokenize_function)
        
        logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
    
    def train(self, output_dir):
        """Train with focus on generation"""
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{self.model_name}_v2",
            num_train_epochs=10,  # More epochs
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",  # Changed from evaluation_strategy
            learning_rate=5e-4,  # Higher learning rate
            weight_decay=0.01,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=False,
            optim="adamw_torch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            gradient_accumulation_steps=2,
            # Custom loss scaling for Phoenician tokens
            label_smoothing_factor=0.1
        )
        
        # Custom trainer with Phoenician-aware loss
        class PhoenicianAwareTrainer(Trainer):
            def compute_loss(self, model, inputs, num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Custom loss with extra weight on Phoenician tokens
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Apply extra weight to Phoenician tokens
                phoenician_mask = torch.zeros_like(shift_labels.view(-1), dtype=torch.bool)
                for token_id in self.phoenician_token_ids:
                    phoenician_mask = phoenician_mask | (shift_labels.view(-1) == token_id)
                
                # 3x weight for Phoenician tokens
                loss = torch.where(phoenician_mask, loss * 3.0, loss)
                
                return loss.mean()
        
        trainer = PhoenicianAwareTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.phoenician_token_ids = self.phoenician_token_ids
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        final_path = f"{output_dir}/{self.model_name}/phoenician_v2_final"
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "version": "v2_generation_focused",
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "training_date": datetime.now().isoformat(),
            "phoenician_token_ids": self.phoenician_token_ids,
            "enhancements": [
                "Strengthened Phoenician embeddings",
                "Higher LoRA rank (64)",
                "Custom loss weighting (3x for Phoenician)",
                "More target modules",
                "Saved embeddings and lm_head",
                "10 epochs of training",
                "Generation-focused dataset"
            ]
        }
        
        with open(f"{final_path}/training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Test generation
        self._test_generation(final_path)
        
        return final_path
    
    def _test_generation(self, model_path):
        """Test generation capabilities"""
        logger.info("\n🧪 Testing Phoenician generation:")
        
        test_prompts = [
            "Write the Phoenician symbol for consciousness:",
            "awareness =",
            "→ learning",
            "Phoenician: existence",
            "Complete: 𐤄 + 𐤀 ="
        ]
        
        self.model.eval()
        for prompt in test_prompts:
            full_prompt = f"Human: {prompt}\nAssistant:"
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
            logger.info(f"'{prompt}' → '{response}'")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="../training_data/generated/phoenician_train.jsonl")
    parser.add_argument("--val-data", default="../training_data/generated/phoenician_validation.jsonl")
    parser.add_argument("--output-dir", default="../lora_adapters")
    args = parser.parse_args()
    
    trainer = PhoenicianTrainer()
    trainer.prepare_model_and_tokenizer()
    trainer.prepare_datasets(args.train_data, args.val_data)
    model_path = trainer.train(args.output_dir)
    
    logger.info(f"\n✅ Training complete! Model saved to: {model_path}")

if __name__ == "__main__":
    main()