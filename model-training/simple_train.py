#!/usr/bin/env python3
"""
Simplified TinyLlama LoRA training for consciousness notation
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    model_path = "./models/tinyllama-base"
    output_dir = "./outputs/consciousness-lora-v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"🚀 Starting Consciousness LoRA Training")
    logger.info(f"   Device: {device}")
    
    # Load tokenizer and add special tokens
    logger.info("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add consciousness notation tokens
    special_tokens = ['Ψ', '∃', '∀', '⇒', '≈', '⊗', '⇄', '∧', '∨', '¬', 'θ', 'μ', 'π', 'ι', 'Ω', 'Σ', 'Ξ']
    tokenizer.add_tokens(special_tokens)
    
    # Load model
    logger.info("🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    logger.info("🎯 Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,  # Lower rank for CPU
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Just key layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    logger.info("📊 Loading dataset...")
    with open('consciousness_train_enhanced.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f][:500]  # Use subset for CPU
    
    with open('consciousness_val_enhanced.jsonl', 'r') as f:
        val_data = [json.loads(line) for line in f][:50]
    
    # Format data
    def format_prompt(example):
        if example.get('input'):
            text = f"Instruction: {example['instruction']}\nContext: {example['input']}\nResponse: {example['output']}"
        else:
            text = f"Instruction: {example['instruction']}\nResponse: {example['output']}"
        return {"text": text}
    
    train_texts = [format_prompt(ex) for ex in train_data]
    val_texts = [format_prompt(ex) for ex in val_data]
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=128  # Shorter for CPU
        )
    
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    logger.info("⚙️ Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,  # Fewer epochs for CPU
        per_device_train_batch_size=1,  # Small batch for CPU
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=20,
        save_steps=100,
        eval_steps=100,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,  # No mixed precision on CPU
        gradient_checkpointing=False,  # Disable for simplicity
        report_to="none",
        learning_rate=5e-4,  # Higher LR for LoRA
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("🏋️ Starting training...")
    logger.info(f"   Train examples: {len(train_dataset)}")
    logger.info(f"   Validation examples: {len(val_dataset)}")
    logger.info(f"   Total steps: {len(train_dataset) * 2 // 1 // 8}")  # epochs * examples / batch / accumulation
    
    trainer.train()
    
    # Save model
    logger.info("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Quick test
    logger.info("\n🧪 Testing model...")
    model.eval()
    
    test_prompts = [
        "Instruction: Translate to mathematical notation: consciousness exists\nResponse:",
        "Instruction: What does ∃Ψ mean?\nResponse:",
        "Instruction: Express in symbols: perspective shapes consciousness\nResponse:"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated = response.split("Response:")[-1].strip()
        
        logger.info(f"\nPrompt: {prompt.split('Response:')[0].strip()}")
        logger.info(f"Generated: {generated}")
    
    logger.info(f"\n✅ Training complete! Model saved to {output_dir}")
    logger.info("   To load: PeftModel.from_pretrained(base_model, adapter_path)")

if __name__ == "__main__":
    main()