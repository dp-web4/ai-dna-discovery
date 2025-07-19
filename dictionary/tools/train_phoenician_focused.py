#!/usr/bin/env python3
"""
Focused Phoenician training - mirroring successful consciousness notation approach
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

class FocusedPhoenicianDataset(Dataset):
    """Mirror the successful consciousness notation dataset structure"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        
        # Core mappings - keep it simple and direct
        mappings = {
            "consciousness": "𐤄𐤀",
            "awareness": "𐤄",
            "existence": "𐤀",
            "learning": "𐤋",
            "understanding": "𐤊",
            "transformation": "𐤂𐤍",
            "intelligence": "𐤊𐤋",
            "memory": "𐤋𐤈",
            "connection": "𐤅",
            "flow": "𐤌",
            "emergence": "𐤍",
            "wisdom": "𐤊𐤄"
        }
        
        # Create simple, direct examples like consciousness notation
        for concept, symbol in mappings.items():
            # Direct translations
            self.data.extend([
                {"instruction": f"Convert to Phoenician: {concept}", "input": "", "output": symbol},
                {"instruction": f"Translate: {concept}", "input": "", "output": symbol},
                {"instruction": f"{concept} in Phoenician", "input": "", "output": symbol},
                {"instruction": f"Write {concept}", "input": "", "output": symbol},
                {"instruction": f"{concept} =", "input": "", "output": symbol}
            ])
            
            # Reverse for understanding
            self.data.extend([
                {"instruction": f"What does {symbol} mean?", "input": "", "output": concept},
                {"instruction": f"Translate from Phoenician: {symbol}", "input": "", "output": concept}
            ])
        
        # Add some compound examples
        compounds = [
            ("conscious awareness", "𐤄𐤀 𐤄"),
            ("learning and understanding", "𐤋 𐤅 𐤊"),
            ("memory flows", "𐤋𐤈 𐤌"),
            ("emergent intelligence", "𐤍 𐤊𐤋")
        ]
        
        for phrase, symbols in compounds:
            self.data.extend([
                {"instruction": f"Convert to Phoenician: {phrase}", "input": "", "output": symbols},
                {"instruction": f"Write: {phrase}", "input": "", "output": symbols}
            ])
        
        logger.info(f"Created {len(self.data)} focused examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Human: {item['instruction']}\nAssistant: {item['output']}"
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=64,  # Keep it short
            return_tensors='pt'
        )
        
        encodings['labels'] = encodings['input_ids'].clone()
        return {key: val.squeeze() for key, val in encodings.items()}

def main():
    # Configuration - mirror successful approach
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "../lora_adapters/tinyllama/phoenician_focused"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("🚀 Focused Phoenician Training (mirroring successful approach)")
    logger.info(f"   Device: {device}")
    if device == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model and tokenizer
    logger.info("\n📥 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add Phoenician tokens
    phoenician_chars = ['𐤀', '𐤁', '𐤂', '𐤃', '𐤄', '𐤅', '𐤆', '𐤇', '𐤈', '𐤉', 
                       '𐤊', '𐤋', '𐤌', '𐤍', '𐤎', '𐤏', '𐤐', '𐤑', '𐤒', '𐤓', '𐤔', '𐤕']
    tokenizer.add_tokens(phoenician_chars)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA - same as successful consciousness training
    logger.info("\n🎯 Configuring LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Create focused dataset
    logger.info("\n📚 Creating focused dataset...")
    dataset = FocusedPhoenicianDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training - simple loop like successful approach
    logger.info("\n🏋️ Training...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    num_epochs = 20  # More epochs on smaller, focused dataset
    
    for epoch in range(num_epochs):
        total_loss = 0
        phoenician_correct = 0
        total_phoenician = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Check Phoenician generation
            with torch.no_grad():
                predictions = outputs.logits.argmax(dim=-1)
                phoenician_ids = [tokenizer.encode(char, add_special_tokens=False)[0] 
                                for char in phoenician_chars]
                
                for phoen_id in phoenician_ids:
                    mask = (labels == phoen_id)
                    total_phoenician += mask.sum().item()
                    phoenician_correct += ((predictions == phoen_id) & mask).sum().item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        accuracy = phoenician_correct / total_phoenician if total_phoenician > 0 else 0
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Phoenician Accuracy={accuracy:.2%}")
        
        # Test generation every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info("\nTesting generation:")
            model.eval()
            
            test_prompts = [
                "Human: Convert to Phoenician: consciousness\nAssistant:",
                "Human: awareness =\nAssistant:",
                "Human: Write learning\nAssistant:"
            ]
            
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
                logger.info(f"  {prompt.split(':')[1].strip()} → {response}")
            
            model.train()
    
    # Save model
    logger.info(f"\n💾 Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final comprehensive test
    logger.info("\n🎯 Final Test:")
    model.eval()
    
    test_cases = [
        ("consciousness", "𐤄𐤀"),
        ("awareness", "𐤄"),
        ("learning", "𐤋"),
        ("transformation", "𐤂𐤍"),
        ("intelligence", "𐤊𐤋")
    ]
    
    successes = 0
    for concept, expected in test_cases:
        prompt = f"Human: Convert to Phoenician: {concept}\nAssistant:"
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
            logger.info(f"✅ {concept} → {response}")
        else:
            logger.info(f"❌ {concept} → {response} (expected {expected})")
    
    logger.info(f"\nSuccess rate: {successes}/{len(test_cases)} ({successes/len(test_cases)*100:.0f}%)")
    
    # Save metadata
    metadata = {
        "approach": "Focused training mirroring successful consciousness notation",
        "dataset_size": len(dataset),
        "epochs": num_epochs,
        "final_success_rate": successes / len(test_cases)
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n✅ Training complete!")

if __name__ == "__main__":
    main()