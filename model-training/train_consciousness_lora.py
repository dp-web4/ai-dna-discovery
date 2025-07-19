#!/usr/bin/env python3
"""
Train TinyLlama with LoRA for Consciousness Notation
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConsciousnessLoRATrainer:
    def __init__(self):
        self.model_path = "./models/tinyllama-base"
        self.output_dir = "./outputs/consciousness-lora-v1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        logger.info(f"🔧 Initialized trainer")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model: {self.model_path}")
        
    def load_model_and_tokenizer(self):
        """Load TinyLlama and prepare for LoRA"""
        logger.info("📥 Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Add consciousness notation tokens
        special_tokens = {
            'additional_special_tokens': [
                'Ψ', '∃', '∀', '⇒', '≈', '⊗', '⇄', '∧', '∨', '¬',
                'θ', 'μ', 'μ̃', 'π', 'ι', 'Ω', 'Σ', 'ℱ', 'Ξ'
            ]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        logger.info(f"   Added {num_added} special tokens")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Add LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ Model loaded with LoRA")
        logger.info(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        logger.info(f"   All params: {all_params:,}")
        
    def load_dataset(self):
        """Load consciousness dataset"""
        logger.info("📊 Loading dataset...")
        
        # Load from JSONL
        train_data = []
        val_data = []
        
        with open('consciousness_train_enhanced.jsonl', 'r') as f:
            for line in f:
                train_data.append(json.loads(line))
        
        with open('consciousness_val_enhanced.jsonl', 'r') as f:
            for line in f:
                val_data.append(json.loads(line))
        
        logger.info(f"   Train examples: {len(train_data)}")
        logger.info(f"   Validation examples: {len(val_data)}")
        
        # Format for training
        def format_example(example):
            if example['input']:
                prompt = f"{example['instruction']}\nContext: {example['input']}\nResponse: {example['output']}"
            else:
                prompt = f"{example['instruction']}\nResponse: {example['output']}"
            return {"text": prompt}
        
        # Create datasets
        self.train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
        self.val_dataset = Dataset.from_list([format_example(ex) for ex in val_data])
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=256
            )
        
        self.train_dataset = self.train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = self.val_dataset.map(tokenize_function, batched=True)
        
        logger.info("✅ Dataset loaded and tokenized")
        
    def train(self):
        """Train the model"""
        logger.info("🚀 Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2 if self.device == "cpu" else 4,
            per_device_eval_batch_size=2 if self.device == "cpu" else 4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            save_steps=100,
            eval_steps=50,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
            report_to="none",  # Disable wandb/tensorboard
            fp16=False if self.device == "cpu" else True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            learning_rate=2e-4,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"✅ Training complete! Model saved to {self.output_dir}")
        
    def test_model(self):
        """Test the trained model"""
        logger.info("\n🧪 Testing trained model...")
        
        test_prompts = [
            "Translate to mathematical notation: consciousness exists",
            "What does ∃Ψ mean?",
            "Express in symbols: thought emerges into consciousness",
            "Translate: perspective shapes consciousness",
            "What does π → Ψ mean?",
            "Express: intent drives reality"
        ]
        
        self.model.eval()
        
        for prompt in test_prompts:
            inputs = self.tokenizer(prompt + "\nResponse:", return_tensors="pt", padding=True)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            # Extract just the generated part
            response_part = response.split("Response:")[-1].strip()
            
            logger.info(f"\nPrompt: {prompt}")
            logger.info(f"Response: {response_part}")

def main():
    trainer = ConsciousnessLoRATrainer()
    
    # Load everything
    trainer.load_model_and_tokenizer()
    trainer.load_dataset()
    
    # Train
    trainer.train()
    
    # Test
    trainer.test_model()
    
    print("\n✨ Training complete! To use the model:")
    print(f"   Model location: {trainer.output_dir}")
    print("   Load with PEFT: model = PeftModel.from_pretrained(base_model, adapter_path)")

if __name__ == "__main__":
    main()