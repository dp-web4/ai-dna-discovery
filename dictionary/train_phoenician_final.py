#!/usr/bin/env python3
"""
Final Phoenician training - exact mirror of train_simple_gpu.py that worked
Using the same dataset format, training approach, and settings
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration - EXACT as train_simple_gpu.py
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "./outputs/phoenician-lora-final"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("🚀 Simple GPU Training for Phoenician Notation")
    logger.info(f"   Device: {device}")
    if device == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model and tokenizer
    logger.info("\n📥 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens - Phoenician
    special_tokens = ['𐤀', '𐤁', '𐤂', '𐤃', '𐤄', '𐤅', '𐤆', '𐤇', '𐤈', '𐤉', 
                     '𐤊', '𐤋', '𐤌', '𐤍', '𐤎', '𐤏', '𐤐', '𐤑', '𐤒', '𐤓', '𐤔', '𐤕', '¬', '∧', '∨']
    tokenizer.add_tokens(special_tokens)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA - EXACT as successful
    logger.info("\n🎯 Configuring LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset - EXACT format as consciousness
    logger.info("\n📊 Loading dataset...")
    with open('phoenician_train_enhanced.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f]
    
    # Format examples - EXACT as consciousness
    def format_example(ex):
        if ex.get('input'):
            return f"### Instruction:\n{ex['instruction']}\n\n### Context:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
    
    train_texts = [format_example(ex) for ex in train_data]
    logger.info(f"   Total examples: {len(train_texts)}")
    
    # Training parameters - EXACT as successful
    batch_size = 4
    learning_rate = 5e-4
    num_epochs = 2
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop - EXACT as successful
    logger.info("\n🏋️ Starting training...")
    model.train()
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        
        # Create progress bar
        pbar = tqdm(range(0, len(train_texts), batch_size), desc=f"Epoch {epoch + 1}")
        
        for i in pbar:
            # Get batch
            batch_texts = train_texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / (len(train_texts) // batch_size)
        logger.info(f"Average loss: {avg_loss:.4f}")
    
    # Save model
    logger.info("\n💾 Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Test the model - EXACT format
    logger.info("\n🧪 Testing trained model...")
    model.eval()
    
    test_prompts = [
        "### Instruction:\nTranslate to mathematical notation: consciousness exists\n\n### Response:\n",
        "### Instruction:\nWhat does 𐤄𐤀 mean?\n\n### Response:\n",
        "### Instruction:\nTranslate: perspective shapes consciousness\n\n### Response:\n",
        "### Instruction:\nExpress: intent drives synchronism\n\n### Response:\n",
        "### Instruction:\nConvert to symbolic form: awareness\n\n### Response:\n",
        "### Instruction:\nEncode as symbols: learning\n\n### Response:\n",
        "### Instruction:\nWhat is the mathematical representation of: memory\n\n### Response:\n"
    ]
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\nTest {i+1}:")
            logger.info(f"Input: {prompt.split('### Response:')[0].strip()}")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            generated = response.split("### Response:\n")[-1].strip()
            logger.info(f"Output: {generated}")
    
    logger.info(f"\n✅ Training complete! Model saved to: {output_dir}")
    
    # Check adapter size
    adapter_files = [f for f in os.listdir(output_dir) if f.startswith("adapter_")]
    if adapter_files:
        total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in adapter_files)
        logger.info(f"   Adapter size: {total_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()