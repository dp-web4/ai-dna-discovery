#!/usr/bin/env python3
"""
Manual training loop for consciousness notation
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

class ConsciousnessDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Format the prompt
        if example.get('input'):
            text = f"Instruction: {example['instruction']}\nContext: {example['input']}\nResponse: {example['output']}"
        else:
            text = f"Instruction: {example['instruction']}\nResponse: {example['output']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = encoding['input_ids'].clone()
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def main():
    print("🚀 Consciousness LoRA Training (Manual)")
    print("=====================================")
    
    # Setup
    model_path = "./models/tinyllama-base"
    output_dir = "./outputs/consciousness-lora-manual"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = ['Ψ', '∃', '∀', '⇒', '≈', '⊗', '⇄', '∧', '∨', '¬', 'θ', 'μ', 'π', 'ι', 'Ω', 'Σ', 'Ξ']
    num_added = tokenizer.add_tokens(special_tokens)
    print(f"   Added {num_added} special tokens")
    
    # Load model
    print("\n🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    print("\n🎯 Adding LoRA...")
    lora_config = LoraConfig(
        r=4,  # Very low rank for CPU
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print("\n📊 Loading dataset...")
    with open('consciousness_train_enhanced.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f][:200]  # Small subset
    
    # Create dataset and dataloader
    train_dataset = ConsciousnessDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    num_epochs = 1
    
    # Training loop
    print(f"\n🏋️ Training for {num_epochs} epoch(s)...")
    print(f"   Examples: {len(train_dataset)}")
    
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(total=len(train_loader), desc="Training")
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'].unsqueeze(0),
                attention_mask=batch['attention_mask'].unsqueeze(0),
                labels=batch['labels'].unsqueeze(0)
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update weights every 4 steps (gradient accumulation)
            if (step + 1) % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Log progress
                if (step + 1) % 20 == 0:
                    avg_loss = total_loss / (step + 1)
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Save model
    print(f"\n💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Test the model
    print("\n🧪 Testing trained model...")
    model.eval()
    
    test_prompts = [
        "Instruction: Translate to mathematical notation: consciousness exists\nResponse:",
        "Instruction: What does ∃Ψ mean?\nResponse:",
        "Instruction: Express in symbols: thought emerges into consciousness\nResponse:"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated = response.split("Response:")[-1].strip()
        
        print(f"\n📝 {prompt.split('Instruction: ')[1].split('\\n')[0]}")
        print(f"   → {generated}")
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Adapter size: ~{os.path.getsize(output_dir + '/adapter_model.safetensors') / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()