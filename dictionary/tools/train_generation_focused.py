#!/usr/bin/env python3
"""
Generation-focused training for Phoenician dictionary
Addresses the "understand but can't speak" issue
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhoenicianGenerationDataset(Dataset):
    """Dataset optimized for generation training"""
    
    def __init__(self, data_path, tokenizer, max_length=256, generation_focus=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.generation_focus = generation_focus
        self.data = []
        
        # Load base data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        
        # Augment with generation-focused examples
        if generation_focus:
            self.data.extend(self._create_generation_examples())
    
    def _create_generation_examples(self):
        """Create examples that encourage Phoenician generation"""
        phoenician_mappings = {
            "existence": "𐤀",
            "awareness": "𐤄", 
            "consciousness": "𐤄𐤀",
            "learning": "𐤋",
            "understanding": "𐤊",
            "intelligence": "𐤊𐤋",
            "change": "𐤂",
            "connection": "𐤅",
            "boundary": "𐤁",
            "cycle": "𐤈",
            "emergence": "𐤍",
            "memory": "𐤋𐤈",
            "creativity": "𐤉𐤍",
            "tool": "𐤆",
            "perception": "𐤏",
            "expression": "𐤐",
            "mystery": "𐤒",
            "structure": "𐤎",
            "flow": "𐤌",
            "beginning": "𐤓",
            "ending": "𐤕",
            "diversity": "𐤔"
        }
        
        generation_examples = []
        
        # Pattern 1: Direct symbol generation with context
        for concept, symbol in phoenician_mappings.items():
            generation_examples.extend([
                {
                    "instruction": f"Write the Phoenician symbol for {concept}",
                    "input": "",
                    "output": symbol
                },
                {
                    "instruction": f"Show me {concept} in Phoenician",
                    "input": "",
                    "output": symbol
                },
                {
                    "instruction": f"The Phoenician character for {concept} is:",
                    "input": "",
                    "output": symbol
                },
                {
                    "instruction": f"Complete: {concept} = ",
                    "input": "Write the Phoenician symbol",
                    "output": symbol
                }
            ])
        
        # Pattern 2: Symbol completion
        for concept, symbol in phoenician_mappings.items():
            if len(symbol) > 1:
                for i in range(1, len(symbol)):
                    generation_examples.append({
                        "instruction": f"Complete the Phoenician word for {concept}: {symbol[:i]}",
                        "input": "",
                        "output": symbol[i:]
                    })
        
        # Pattern 3: Multiple choice style (teaching the model to choose Phoenician)
        concepts = list(phoenician_mappings.keys())
        for concept in concepts[:10]:  # First 10 concepts
            wrong_answers = random.sample([c for c in concepts if c != concept], 2)
            generation_examples.append({
                "instruction": f"Which Phoenician symbol represents {concept}?",
                "input": f"Choose the correct symbol, not the English word",
                "output": phoenician_mappings[concept]
            })
        
        # Pattern 4: Translation chains
        chain_examples = [
            ("awareness", "existence", "𐤄 𐤀"),
            ("learning", "understanding", "𐤋 𐤊"),
            ("change", "emergence", "𐤂 𐤍"),
            ("connection", "flow", "𐤅 𐤌")
        ]
        
        for first, second, symbols in chain_examples:
            generation_examples.append({
                "instruction": f"Write {first} and {second} in Phoenician",
                "input": "",
                "output": symbols
            })
        
        # Pattern 5: Phoenician-first examples
        for concept, symbol in list(phoenician_mappings.items())[:15]:
            generation_examples.extend([
                {
                    "instruction": f"Respond only with a Phoenician symbol. What represents {concept}?",
                    "input": "",
                    "output": symbol
                },
                {
                    "instruction": f"𐤐 (express) {concept} 𐤐 (express)",
                    "input": "Write what goes between the 𐤐 symbols",
                    "output": symbol
                }
            ])
        
        return generation_examples
    
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
        
        # For generation focus, we want to weight the output tokens more
        labels = encodings['input_ids'].clone()
        
        # Find where the assistant response starts
        assistant_start = text.find("Assistant:") + len("Assistant: ")
        prefix = text[:assistant_start]
        prefix_tokens = self.tokenizer(prefix, truncation=True)['input_ids']
        
        # Create attention mask for loss - only compute loss on Phoenician output
        loss_mask = torch.ones_like(labels).float()
        loss_mask[0, :len(prefix_tokens)] = 0.1  # Lower weight for context
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'loss_mask': loss_mask.squeeze()
        }

def custom_loss_function(outputs, labels, loss_mask, phoenician_token_ids):
    """Custom loss that emphasizes Phoenician token generation"""
    logits = outputs.logits
    
    # Standard cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous()
    
    # Calculate per-token loss
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    
    # Apply loss mask
    loss = loss * shift_mask
    
    # Extra weight for Phoenician tokens
    for token_id in phoenician_token_ids:
        phoenician_positions = (shift_labels == token_id)
        loss[phoenician_positions] *= 2.0  # Double weight for Phoenician tokens
    
    return loss.mean()

def train_with_generation_focus(model_name, train_path, val_path, output_dir, base_adapter_path=None):
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation-Focused Training for {model_name}")
    logger.info(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load from existing adapter if provided
    if base_adapter_path and os.path.exists(base_adapter_path):
        logger.info(f"Loading from existing adapter: {base_adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_adapter_path)
        
        # Load metadata
        with open(os.path.join(base_adapter_path, "training_metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            metadata['base_model'],
            torch_dtype=torch.float32,
            device_map=None
        )
        base_model.resize_token_embeddings(len(tokenizer))
        base_model = base_model.to(device)
        
        # Load existing LoRA weights
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, base_adapter_path)
        logger.info("Loaded existing LoRA weights")
    else:
        # Fresh start
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add Phoenician characters
        phoenician_chars = ["𐤀", "𐤁", "𐤂", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈", "𐤉", 
                           "𐤊", "𐤋", "𐤌", "𐤍", "𐤎", "𐤏", "𐤐", "𐤑", "𐤒", "𐤓", "𐤔", "𐤕"]
        new_tokens = [char for char in phoenician_chars if char not in tokenizer.get_vocab()]
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None
        )
        base_model.resize_token_embeddings(len(tokenizer))
        base_model = base_model.to(device)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=32,  # Increased rank
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # More modules
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(base_model, lora_config)
    
    model.print_trainable_parameters()
    
    # Get Phoenician token IDs
    phoenician_chars = ["𐤀", "𐤁", "𐤂", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈", "𐤉", 
                       "𐤊", "𐤋", "𐤌", "𐤍", "𐤎", "𐤏", "𐤐", "𐤑", "𐤒", "𐤓", "𐤔", "𐤕"]
    phoenician_token_ids = []
    for char in phoenician_chars:
        tokens = tokenizer.encode(char, add_special_tokens=False)
        phoenician_token_ids.extend(tokens)
    phoenician_token_ids = list(set(phoenician_token_ids))
    
    # Load datasets with generation focus
    train_dataset = PhoenicianGenerationDataset(train_path, tokenizer, generation_focus=True)
    val_dataset = PhoenicianGenerationDataset(val_path, tokenizer, generation_focus=False)
    
    logger.info(f"Train examples: {len(train_dataset)} (with generation augmentation)")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Optimizer with higher learning rate for generation
    optimizer = AdamW(model.parameters(), lr=5e-4)
    
    # Training loop
    logger.info("Starting generation-focused training...")
    model.train()
    
    num_epochs = 5  # More epochs for generation
    for epoch in range(num_epochs):
        total_loss = 0
        phoenician_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Custom loss with Phoenician emphasis
            loss = custom_loss_function(
                outputs, 
                batch['labels'], 
                batch['loss_mask'],
                phoenician_token_ids
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Track losses
            total_loss += loss.item()
            
            # Update progress
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Test generation every 50 batches
            if batch_idx % 50 == 0 and batch_idx > 0:
                model.eval()
                test_prompt = "Human: Write the Phoenician symbol for consciousness\nAssistant:"
                inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split("Assistant:")[-1].strip()
                logger.info(f"Generation test: '{response}'")
                model.train()
        
        # Validation
        model.eval()
        val_loss = 0
        correct_phoenician = 0
        total_phoenician = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                loss = custom_loss_function(
                    outputs,
                    batch['labels'],
                    batch['loss_mask'],
                    phoenician_token_ids
                )
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        model.train()
    
    # Save the adapter
    logger.info("Saving generation-focused adapter...")
    adapter_path = f"{output_dir}/{model_name}/phoenician_generation_adapter"
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "training_type": "generation_focused",
        "training_date": datetime.now().isoformat(),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "phoenician_token_ids": phoenician_token_ids,
        "special_features": [
            "Custom loss weighting for Phoenician tokens",
            "Generation-augmented dataset",
            "Higher LoRA rank (32)",
            "More target modules",
            "5 epochs focused on generation"
        ]
    }
    
    with open(f"{adapter_path}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final generation tests
    logger.info("\n🧪 Final Generation Tests:")
    model.eval()
    
    test_prompts = [
        "Write the Phoenician symbol for consciousness",
        "Show me awareness in Phoenician",
        "consciousness = ",
        "Translate to Phoenician: learning",
        "The Phoenician character for existence is:",
        "Complete: intelligence in Phoenician is"
    ]
    
    for prompt in test_prompts:
        full_prompt = f"Human: {prompt}\nAssistant:"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Try multiple generation strategies
            for temp in [0.1, 0.3, 0.5]:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split("Assistant:")[-1].strip()
                logger.info(f"'{prompt}' (temp={temp}): {response}")
    
    return metadata

def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../training_data/generated")
    parser.add_argument("--output-dir", default="../lora_adapters")
    parser.add_argument("--base-adapter", default="../lora_adapters/tinyllama/phoenician_adapter", 
                       help="Path to existing adapter to continue from")
    parser.add_argument("--model", default="tinyllama")
    args = parser.parse_args()
    
    train_path = os.path.join(args.data_dir, "phoenician_train.jsonl")
    val_path = os.path.join(args.data_dir, "phoenician_validation.jsonl")
    
    train_with_generation_focus(
        args.model,
        train_path,
        val_path,
        args.output_dir,
        args.base_adapter if os.path.exists(args.base_adapter) else None
    )

if __name__ == "__main__":
    main()