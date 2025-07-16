#!/usr/bin/env python3
"""
Hidden State Persistence Experiment
Test if we can maintain conversation state through hidden state manipulation
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
from datetime import datetime

class HiddenStateMemory:
    """Maintain hidden states between model calls"""
    
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        print(f"Loading {model_name} with hidden state access...")
        
        # Load model with hidden state output
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            output_hidden_states=True,
            return_dict=True
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_memory = None
        self.conversation_cache = None
        
    def get_hidden_states(self, text):
        """Extract hidden states from model"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # outputs.hidden_states is tuple of (num_layers + 1) tensors
        # Each tensor shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.hidden_states
        
        # Get final hidden state from last layer
        final_hidden = hidden_states[-1]
        
        # Also get intermediate layer states for analysis
        layer_states = {
            'layer_0': hidden_states[0].cpu().numpy(),
            'layer_16': hidden_states[16].cpu().numpy() if len(hidden_states) > 16 else None,
            'layer_final': final_hidden.cpu().numpy()
        }
        
        return final_hidden, layer_states
    
    def generate_with_hidden_influence(self, prompt, temperature=0.7, use_memory=False):
        """Generate text with optional hidden state influence"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # If we have memory and want to use it
        if use_memory and self.hidden_memory is not None:
            # This is experimental - influence initial hidden states
            # Note: This is a simplified approach; real implementation would be more complex
            print("  [Using hidden state memory]")
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Extract hidden states from generation
        if hasattr(outputs, 'hidden_states'):
            # Store for next interaction
            self.hidden_memory = outputs.hidden_states
            
        return response
    
    def analyze_state_persistence(self):
        """Test if hidden states show memory-like properties"""
        print("\n=== HIDDEN STATE PERSISTENCE EXPERIMENT ===\n")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }
        
        # Test 1: Baseline - repeated queries without context
        print("Test 1: Baseline (no memory)")
        prompt1 = "My name is Alice and I love"
        
        response1a = self.generate_with_hidden_influence(prompt1, temperature=0)
        hidden1a, layers1a = self.get_hidden_states(prompt1 + response1a)
        
        time.sleep(2)
        
        response1b = self.generate_with_hidden_influence(prompt1, temperature=0)
        hidden1b, layers1b = self.get_hidden_states(prompt1 + response1b)
        
        # Compare hidden states
        similarity = torch.cosine_similarity(
            hidden1a.mean(dim=1), 
            hidden1b.mean(dim=1)
        ).item()
        
        results['tests'].append({
            'name': 'baseline_repetition',
            'prompt': prompt1,
            'response_1': response1a.replace(prompt1, '').strip(),
            'response_2': response1b.replace(prompt1, '').strip(),
            'hidden_similarity': similarity
        })
        
        print(f"  Response 1: {response1a.replace(prompt1, '').strip()}")
        print(f"  Response 2: {response1b.replace(prompt1, '').strip()}")
        print(f"  Hidden state similarity: {similarity:.4f}")
        
        # Test 2: Context accumulation
        print("\n\nTest 2: Context accumulation")
        
        # First exchange
        prompt2a = "My favorite color is blue."
        response2a = self.generate_with_hidden_influence(prompt2a, temperature=0)
        hidden2a, _ = self.get_hidden_states(prompt2a + response2a)
        
        # Second exchange with context
        prompt2b = "What is my favorite color?"
        full_context = f"{prompt2a} {response2a}\n{prompt2b}"
        response2b = self.generate_with_hidden_influence(full_context, temperature=0)
        
        # Third exchange without context (testing if model "remembers")
        response2c = self.generate_with_hidden_influence(prompt2b, temperature=0)
        
        results['tests'].append({
            'name': 'context_test',
            'exchange_1': {'prompt': prompt2a, 'response': response2a.replace(prompt2a, '').strip()},
            'with_context': {'prompt': prompt2b, 'response': response2b.replace(full_context, '').strip()},
            'without_context': {'prompt': prompt2b, 'response': response2c.replace(prompt2b, '').strip()}
        })
        
        print(f"  Setup: {prompt2a}")
        print(f"  Query with context: {response2b.replace(full_context, '').strip()}")
        print(f"  Query without context: {response2c.replace(prompt2b, '').strip()}")
        
        # Test 3: Hidden state analysis across layers
        print("\n\nTest 3: Hidden state evolution")
        prompt3 = "The meaning of life is"
        
        _, layers = self.get_hidden_states(prompt3)
        
        # Analyze how hidden states change across layers
        layer_stats = {}
        for layer_name, layer_state in layers.items():
            if layer_state is not None:
                layer_stats[layer_name] = {
                    'mean': float(np.mean(layer_state)),
                    'std': float(np.std(layer_state)),
                    'shape': layer_state.shape
                }
        
        results['tests'].append({
            'name': 'layer_evolution',
            'prompt': prompt3,
            'layer_statistics': layer_stats
        })
        
        print(f"  Analyzing hidden states for: '{prompt3}'")
        for layer, stats in layer_stats.items():
            print(f"    {layer}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        # Save results
        with open('/home/dp/ai-workspace/ai-agents/hidden_state_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def experimental_state_injection(self):
        """Experimental: Try to inject state from one context into another"""
        print("\n\n=== EXPERIMENTAL STATE INJECTION ===\n")
        
        # Get hidden states from Context A
        context_a = "I am an expert chef who specializes in French cuisine."
        response_a = self.generate_with_hidden_influence(context_a, temperature=0)
        hidden_a, _ = self.get_hidden_states(context_a)
        
        print(f"Context A: {context_a}")
        print(f"Response A: {response_a.replace(context_a, '').strip()}")
        
        # Get hidden states from Context B  
        context_b = "I am a novice cook who just started learning."
        response_b = self.generate_with_hidden_influence(context_b, temperature=0)
        hidden_b, _ = self.get_hidden_states(context_b)
        
        print(f"\nContext B: {context_b}")
        print(f"Response B: {response_b.replace(context_b, '').strip()}")
        
        # Now ask a question and see which "personality" emerges
        question = "What's your best cooking tip?"
        
        print(f"\nQuestion: {question}")
        print("Response (neutral):", self.generate_with_hidden_influence(question, temperature=0).replace(question, '').strip())
        
        # Calculate which hidden state the model is "closer" to
        hidden_q, _ = self.get_hidden_states(question)
        
        sim_to_a = torch.cosine_similarity(hidden_q.mean(dim=1), hidden_a.mean(dim=1)).item()
        sim_to_b = torch.cosine_similarity(hidden_q.mean(dim=1), hidden_b.mean(dim=1)).item()
        
        print(f"\nHidden state similarities:")
        print(f"  To expert chef: {sim_to_a:.4f}")
        print(f"  To novice cook: {sim_to_b:.4f}")


def main():
    # Initialize with hidden state tracking
    memory = HiddenStateMemory()
    
    # Run persistence analysis
    results = memory.analyze_state_persistence()
    
    # Run experimental state injection
    memory.experimental_state_injection()
    
    print("\n\n=== EXPERIMENT COMPLETE ===")
    print("Results saved to hidden_state_results.json")
    
    # Theoretical next steps
    print("\n\nNext steps for true hidden state memory:")
    print("1. Implement KV-cache persistence between calls")
    print("2. Create hidden state 'checkpoints' for conversation branches")
    print("3. Develop state merging algorithms for multiple contexts")
    print("4. Build attention-mask manipulation for selective memory")


if __name__ == "__main__":
    main()