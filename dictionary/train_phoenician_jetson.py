#!/usr/bin/env python3
"""
Phoenician training optimized for Jetson Orin Nano
Reduced memory footprint while maintaining effectiveness
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
    # Configuration optimized for Jetson
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "./outputs/phoenician-lora-jetson"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("🚀 Phoenician Training on Jetson Orin Nano")
    logger.info(f"   Device: {device}")
    if device == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        # Log memory info
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Load model and tokenizer
    logger.info("\n📥 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add Phoenician tokens
    special_tokens = ['𐤀', '𐤁', '𐤂', '𐤃', '𐤄', '𐤅', '𐤆', '𐤇', '𐤈', '𐤉', 
                     '𐤊', '𐤋', '𐤌', '𐤍', '𐤎', '𐤏', '𐤐', '𐤑', '𐤒', '𐤓', '𐤔', '𐤕', '¬', '∧', '∨']
    tokenizer.add_tokens(special_tokens)
    
    # Load with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True  # Important for Jetson
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
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
    
    # Load dataset
    logger.info("\n📊 Loading dataset...")
    with open('phoenician_train_enhanced.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f]
    
    # Format examples
    def format_example(ex):
        if ex.get('input'):
            return f"### Instruction:\n{ex['instruction']}\n\n### Context:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
    
    train_texts = [format_example(ex) for ex in train_data]
    logger.info(f"   Total examples: {len(train_texts)}")
    
    # Training parameters optimized for Jetson
    batch_size = 2  # Reduced for memory
    learning_rate = 5e-4
    num_epochs = 2
    max_length = 256  # Reduced for memory
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop with memory management
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
            
            # Tokenize with reduced max_length
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
            
            # Update metrics
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / (len(train_texts) // batch_size)
        logger.info(f"Average loss: {avg_loss:.4f}")
        
        # Clear cache after epoch
        torch.cuda.empty_cache()
    
    # Save model
    logger.info("\n💾 Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Test the model
    logger.info("\n🧪 Testing trained model...")
    model.eval()
    
    test_prompts = [
        "### Instruction:\nConvert to symbolic form: awareness\n\n### Response:\n",
        "### Instruction:\nWhat does 𐤄𐤀 mean?\n\n### Response:\n",
        "### Instruction:\nEncode as symbols: learning\n\n### Response:\n"
    ]
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\nTest {i+1}:")
            logger.info(f"Input: {prompt.split('### Response:')[0].strip()}")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            generated = response.split("### Response:\n")[-1].strip()
            logger.info(f"Output: {generated}")
    
    logger.info(f"\n✅ Training complete on Jetson! Model saved to: {output_dir}")

if __name__ == "__main__":
    main()