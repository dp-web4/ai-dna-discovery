#!/usr/bin/env python3
"""
Analyze barriers to Phoenician generation
Understanding the "comprehension vs production" gap
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np

def analyze_generation_barriers(adapter_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Get Phoenician token IDs
    phoenician_chars = ["𐤀", "𐤁", "𐤂", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈", "𐤉", 
                       "𐤊", "𐤋", "𐤌", "𐤍", "𐤎", "𐤏", "𐤐", "𐤑", "𐤒", "𐤓", "𐤔", "𐤕"]
    
    phoenician_token_info = {}
    for char in phoenician_chars:
        tokens = tokenizer.encode(char, add_special_tokens=False)
        phoenician_token_info[char] = tokens[0] if tokens else None
    
    print("🔍 Analyzing Generation Barriers\n")
    
    # 1. Token Embedding Analysis
    print("1️⃣ Token Embedding Analysis:")
    embeddings = model.get_input_embeddings().weight.data
    
    # Compare Phoenician vs regular token embeddings
    phoenician_ids = [id for id in phoenician_token_info.values() if id is not None]
    regular_ids = list(range(100, 1000))  # Sample of regular tokens
    
    phoenician_norms = [embeddings[id].norm().item() for id in phoenician_ids]
    regular_norms = [embeddings[id].norm().item() for id in regular_ids]
    
    print(f"   Phoenician embedding norms: mean={np.mean(phoenician_norms):.3f}, std={np.std(phoenician_norms):.3f}")
    print(f"   Regular embedding norms: mean={np.mean(regular_norms):.3f}, std={np.std(regular_norms):.3f}")
    print(f"   Ratio: {np.mean(phoenician_norms) / np.mean(regular_norms):.3f}")
    
    # 2. Output Layer Analysis
    print("\n2️⃣ Output Layer Bias Analysis:")
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
        if hasattr(lm_head, 'bias') and lm_head.bias is not None:
            phoenician_biases = [lm_head.bias[id].item() for id in phoenician_ids]
            regular_biases = [lm_head.bias[id].item() for id in regular_ids]
            print(f"   Phoenician biases: mean={np.mean(phoenician_biases):.3f}")
            print(f"   Regular biases: mean={np.mean(regular_biases):.3f}")
        else:
            print("   No output bias found")
    
    # 3. Generation Probability Analysis
    print("\n3️⃣ Generation Probability Analysis:")
    
    test_contexts = [
        "Human: Write the Phoenician symbol for consciousness\nAssistant:",
        "Human: Translate to Phoenician: awareness\nAssistant:",
        "Human: Show me existence in Phoenician\nAssistant:",
        "Human: consciousness = \nAssistant:",
        "Human: The Phoenician character for learning is\nAssistant:"
    ]
    
    prob_data = []
    
    for context in test_contexts:
        inputs = tokenizer(context, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            # Get probabilities for Phoenician tokens
            phoenician_probs = {}
            for char, token_id in phoenician_token_info.items():
                if token_id is not None:
                    phoenician_probs[char] = probs[token_id].item()
            
            # Get top 10 token probabilities
            top_probs, top_indices = torch.topk(probs, 10)
            
            prob_data.append({
                'context': context.split('\n')[-2],  # Just the question part
                'phoenician_probs': phoenician_probs,
                'top_tokens': [(tokenizer.decode([idx]), prob.item()) for idx, prob in zip(top_indices, top_probs)]
            })
    
    # Display results
    for data in prob_data:
        print(f"\nContext: {data['context']}")
        print("Top Phoenician probabilities:")
        sorted_phoenician = sorted(data['phoenician_probs'].items(), key=lambda x: x[1], reverse=True)[:5]
        for char, prob in sorted_phoenician:
            print(f"   {char}: {prob:.6f}")
        print("Top overall tokens:")
        for token, prob in data['top_tokens'][:5]:
            print(f"   '{token}': {prob:.6f}")
    
    # 4. Attention Pattern Analysis
    print("\n4️⃣ Attention Pattern Analysis:")
    
    # Check if model attends to Phoenician characters in input
    test_input = "Human: What does 𐤄𐤀 mean?\nAssistant:"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # List of attention tensors
        
        # Find positions of Phoenician tokens
        tokens = tokenizer.encode(test_input)
        phoenician_positions = []
        for i, token in enumerate(tokens):
            if token in phoenician_ids:
                phoenician_positions.append(i)
        
        if phoenician_positions and attentions:
            # Look at last layer attention to Phoenician tokens
            last_attention = attentions[-1][0].mean(dim=0)  # Average over heads
            phoenician_attention = last_attention[:, phoenician_positions].mean().item()
            overall_attention = last_attention.mean().item()
            
            print(f"   Average attention to Phoenician tokens: {phoenician_attention:.4f}")
            print(f"   Overall average attention: {overall_attention:.4f}")
            print(f"   Ratio: {phoenician_attention / overall_attention:.2f}x")
    
    # 5. Hypothesis Summary
    print("\n5️⃣ Hypothesis Summary:")
    print("\nPossible barriers to generation:")
    print("1. Token frequency: Phoenician tokens are rare in training data")
    print("2. Embedding initialization: New tokens may have weak embeddings")
    print("3. Output bias: Model may be biased toward common tokens")
    print("4. Attention patterns: Model may not associate contexts with Phoenician output")
    print("5. Training objective: Standard LM loss may not emphasize rare tokens enough")
    
    return prob_data

def create_visualization(prob_data, output_path="phoenician_generation_analysis.txt"):
    """Create text visualization of generation probabilities"""
    
    print("\n📊 Generation Probability Summary:")
    print("="*60)
    
    # Phoenician token probabilities across contexts
    phoenician_chars = ["𐤀", "𐤄", "𐤊", "𐤋", "𐤂"]  # Top 5 chars
    
    for data in prob_data:
        print(f"\nContext: {data['context']}")
        print("Phoenician probabilities:")
        for char in phoenician_chars:
            prob = data['phoenician_probs'].get(char, 0)
            bar = "█" * int(prob * 1000)  # Scale for visibility
            print(f"  {char}: {prob:.6f} {bar}")
    
    # Average probabilities summary
    all_phoenician_probs = []
    for data in prob_data:
        all_phoenician_probs.extend(data['phoenician_probs'].values())
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Average Phoenician probability: {np.mean(all_phoenician_probs):.6f}")
    print(f"   Max Phoenician probability: {np.max(all_phoenician_probs):.6f}")
    print(f"   Min Phoenician probability: {np.min(all_phoenician_probs):.6f}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="../lora_adapters/tinyllama/phoenician_adapter")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    prob_data = analyze_generation_barriers(args.adapter)
    
    if args.visualize:
        create_visualization(prob_data)

if __name__ == "__main__":
    main()