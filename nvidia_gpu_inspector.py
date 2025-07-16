#!/usr/bin/env python3
"""
Use NVIDIA tools to inspect GPU memory and detect changes
Focuses on what we CAN access without modifying Ollama
"""

import subprocess
import time
import json
import hashlib
from datetime import datetime
import ollama

class NvidiaGPUInspector:
    def __init__(self):
        self.measurements = []
        
    def get_detailed_gpu_state(self):
        """Get comprehensive GPU state using nvidia-smi"""
        state = {}
        
        # Memory info
        result = subprocess.run([
            "nvidia-smi", 
            "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu,utilization.memory,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True)
        
        values = result.stdout.strip().split(', ')
        state['memory_used_mb'] = int(values[0])
        state['memory_free_mb'] = int(values[1])
        state['memory_total_mb'] = int(values[2])
        state['gpu_utilization'] = int(values[3])
        state['memory_utilization'] = int(values[4])
        state['temperature'] = int(values[5])
        
        # Process info
        result = subprocess.run([
            "nvidia-smi", 
            "pmon", "-c", "1"
        ], capture_output=True, text=True)
        
        state['processes'] = []
        for line in result.stdout.strip().split('\n'):
            if 'ollama' in line:
                parts = line.split()
                if len(parts) >= 7:
                    state['processes'].append({
                        'pid': parts[1],
                        'sm%': parts[3],
                        'mem%': parts[4],
                        'enc%': parts[5],
                        'dec%': parts[6]
                    })
        
        # Create a hash of the state for comparison
        state_str = json.dumps(state, sort_keys=True)
        state['state_hash'] = hashlib.md5(state_str.encode()).hexdigest()[:8]
        
        return state
    
    def test_runtime_weight_stability(self, model_name: str = "deepseek-coder:1.3b"):
        """
        Test if weights remain stable during runtime
        Using multiple detection methods
        """
        
        print(f"Testing Runtime Weight Stability for {model_name}")
        print("="*60)
        
        # Ensure model is loaded
        print("\n1. Loading model into GPU...")
        ollama.generate(model=model_name, prompt="", keep_alive="24h")
        time.sleep(2)
        
        # Get baseline GPU state
        baseline_gpu = self.get_detailed_gpu_state()
        print(f"   Memory used: {baseline_gpu['memory_used_mb']} MB")
        
        # Test 1: Deterministic response consistency
        print("\n2. Testing deterministic response consistency...")
        
        test_cases = [
            {
                'prompt': 'Complete this function: def factorial(n):',
                'options': {'temperature': 0.0, 'seed': 12345, 'num_predict': 30}
            },
            {
                'prompt': 'The meaning of recursion is',
                'options': {'temperature': 0.0, 'seed': 54321, 'num_predict': 20}
            },
            {
                'prompt': '2 + 2 =',
                'options': {'temperature': 0.0, 'seed': 11111, 'num_predict': 5}
            }
        ]
        
        baseline_responses = {}
        for i, test in enumerate(test_cases):
            resp = ollama.generate(
                model=model_name,
                prompt=test['prompt'],
                options=test['options'],
                keep_alive="24h"
            )
            baseline_responses[i] = resp['response']
            print(f"   Test {i+1}: {test['prompt'][:30]}... -> {len(resp['response'])} chars")
        
        # Test 2: Heavy usage phase
        print("\n3. Heavy usage phase (50 diverse prompts)...")
        
        diverse_prompts = [
            "Write a sorting algorithm",
            "Explain quantum computing",
            "def binary_search(arr, target):",
            "What is machine learning?",
            "class NeuralNetwork:",
            "Recursive fibonacci implementation",
            "Database optimization techniques",
            "REST API best practices",
            "Microservices architecture",
            "Docker container management"
        ] * 5  # 50 total
        
        for i, prompt in enumerate(diverse_prompts):
            ollama.generate(
                model=model_name,
                prompt=prompt,
                options={'temperature': 0.8, 'num_predict': 50},
                keep_alive="24h"
            )
            if i % 10 == 0:
                print(f"   Progress: {i}/50")
                
                # Check GPU state
                current_gpu = self.get_detailed_gpu_state()
                memory_delta = current_gpu['memory_used_mb'] - baseline_gpu['memory_used_mb']
                if abs(memory_delta) > 100:  # 100MB threshold
                    print(f"   ⚠️  Memory change detected: {memory_delta:+d} MB")
        
        # Test 3: Re-check deterministic responses
        print("\n4. Re-checking deterministic responses...")
        
        changes_detected = []
        for i, test in enumerate(test_cases):
            resp = ollama.generate(
                model=model_name,
                prompt=test['prompt'],
                options=test['options'],
                keep_alive="24h"
            )
            
            if resp['response'] != baseline_responses[i]:
                changes_detected.append({
                    'test': i,
                    'prompt': test['prompt'],
                    'baseline_len': len(baseline_responses[i]),
                    'new_len': len(resp['response']),
                    'identical': False
                })
                print(f"   ⚠️  Test {i+1}: RESPONSE CHANGED!")
            else:
                print(f"   ✓ Test {i+1}: Identical response")
        
        # Test 4: Embedding stability
        print("\n5. Testing embedding stability...")
        
        embedding_test_prompts = ["hello", "world", "recursive", "function", "model"]
        baseline_embeddings = {}
        
        for prompt in embedding_test_prompts:
            emb = ollama.embeddings(model=model_name, prompt=prompt)
            baseline_embeddings[prompt] = emb['embedding'][:50]  # First 50 dims
        
        # Heavy usage
        for _ in range(20):
            ollama.generate(
                model=model_name,
                prompt="Complex recursive algorithm implementation",
                options={'temperature': 0.9},
                keep_alive="24h"
            )
        
        # Re-check embeddings
        embedding_drifts = []
        for prompt in embedding_test_prompts:
            emb = ollama.embeddings(model=model_name, prompt=prompt)
            new_embedding = emb['embedding'][:50]
            
            # Calculate drift
            import numpy as np
            drift = np.linalg.norm(
                np.array(baseline_embeddings[prompt]) - np.array(new_embedding)
            )
            embedding_drifts.append(drift)
            
            if drift > 0.0001:
                print(f"   ⚠️  Embedding drift for '{prompt}': {drift:.8f}")
        
        # Final GPU state
        final_gpu = self.get_detailed_gpu_state()
        
        # Analysis
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'deterministic_changes': len(changes_detected),
            'max_embedding_drift': max(embedding_drifts) if embedding_drifts else 0,
            'memory_change_mb': final_gpu['memory_used_mb'] - baseline_gpu['memory_used_mb'],
            'changes_detected': changes_detected,
            'verdict': 'STATIC'  # Will update based on findings
        }
        
        if len(changes_detected) > 0:
            print(f"\n⚠️  RUNTIME CHANGES DETECTED!")
            print(f"- {len(changes_detected)} deterministic responses changed")
            result['verdict'] = 'DYNAMIC'
        
        if max(embedding_drifts) > 0.0001:
            print(f"- Maximum embedding drift: {max(embedding_drifts):.8f}")
            result['verdict'] = 'DYNAMIC'
        
        if abs(result['memory_change_mb']) > 100:
            print(f"- GPU memory change: {result['memory_change_mb']:+d} MB")
            result['verdict'] = 'POSSIBLE_DYNAMIC'
        
        if result['verdict'] == 'STATIC':
            print("\n✓ NO RUNTIME PLASTICITY DETECTED")
            print("Model weights appear to remain static during inference")
        
        # Save results
        with open('nvidia_gpu_inspection_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to nvidia_gpu_inspection_results.json")
        
        return result

if __name__ == "__main__":
    inspector = NvidiaGPUInspector()
    
    # First, let's clean up GPU
    print("Preparing clean GPU state...")
    subprocess.run(["nvidia-smi"], capture_output=True)
    
    # Run the test
    results = inspector.test_runtime_weight_stability()