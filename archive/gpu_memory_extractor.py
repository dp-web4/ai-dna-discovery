#!/usr/bin/env python3
"""
Extract and analyze model weights from GPU memory at runtime
This is the real test of runtime plasticity
"""

import os
import subprocess
import struct
import numpy as np
import time
import json
from datetime import datetime

class GPUMemoryExtractor:
    def __init__(self):
        self.results = {
            "theory": "Extracting runtime weights from GPU to test plasticity",
            "measurements": []
        }
        
    def get_gpu_memory_info(self):
        """Get current GPU memory usage and process info"""
        try:
            # Get GPU memory info
            result = subprocess.run([
                "nvidia-smi", 
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True)
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line and 'ollama' in line:
                    parts = line.split(', ')
                    processes.append({
                        'pid': int(parts[0]),
                        'name': parts[1],
                        'memory_mb': int(parts[2])
                    })
            
            return processes
            
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return []
    
    def dump_gpu_memory_via_gdb(self, pid: int, output_file: str):
        """
        Use GDB to dump GPU memory from running process
        This requires sudo access
        """
        print(f"Attempting to dump GPU memory from PID {pid}...")
        
        # GDB commands to dump memory regions
        gdb_commands = f"""
attach {pid}
info proc mappings
# Look for GPU memory regions (usually large, aligned regions)
# This is a simplified version - real implementation would parse mappings
maintenance info sections
detach
quit
"""
        
        try:
            # Write GDB commands to temp file
            with open('/tmp/gdb_commands.txt', 'w') as f:
                f.write(gdb_commands)
            
            # Run GDB (requires sudo)
            result = subprocess.run([
                'sudo', 'gdb', '-batch', '-x', '/tmp/gdb_commands.txt'
            ], capture_output=True, text=True)
            
            print("GDB output:")
            print(result.stdout[:500])  # First 500 chars
            
            return result.stdout
            
        except Exception as e:
            print(f"GDB approach failed: {e}")
            return None
    
    def use_cuda_memcpy(self):
        """
        Alternative approach using CUDA tools to copy GPU memory
        """
        print("\nTrying CUDA approach...")
        
        cuda_code = """
#include <cuda_runtime.h>
#include <stdio.h>

// This would need to hook into the running Ollama process
// to find the actual memory addresses of the model weights
int main() {
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    printf("GPU memory: free=%zu MB, total=%zu MB\\n", 
           free_byte/1048576, total_byte/1048576);
    
    // In reality, we'd need to:
    // 1. Find the CUDA context of the Ollama process
    // 2. Locate the model weight tensors
    // 3. Copy them to host memory
    // This requires deep integration with Ollama's internals
    
    return 0;
}
"""
        
        # Save and try to compile
        with open('/tmp/cuda_mem_test.cu', 'w') as f:
            f.write(cuda_code)
        
        try:
            # Try to compile (requires CUDA toolkit)
            subprocess.run(['nvcc', '/tmp/cuda_mem_test.cu', '-o', '/tmp/cuda_mem_test'])
            result = subprocess.run(['/tmp/cuda_mem_test'], capture_output=True, text=True)
            print(result.stdout)
        except:
            print("CUDA toolkit not available or compilation failed")
    
    def extract_via_pytorch_hooks(self):
        """
        More practical approach: Use PyTorch hooks if we can load the model
        """
        print("\nTrying PyTorch hooks approach...")
        
        try:
            import torch
            import torch.nn as nn
            
            # This approach would work if we could load the actual model
            # For now, demonstrate the concept
            
            class WeightMonitor:
                def __init__(self):
                    self.weight_snapshots = []
                    self.activation_snapshots = []
                
                def register_hooks(self, model):
                    """Register forward hooks to monitor weights during inference"""
                    def make_hook(name):
                        def hook(module, input, output):
                            # Capture weight state during forward pass
                            if hasattr(module, 'weight'):
                                weight_copy = module.weight.detach().cpu().numpy().copy()
                                self.weight_snapshots.append({
                                    'name': name,
                                    'timestamp': time.time(),
                                    'weight_hash': hash(weight_copy.tobytes()),
                                    'weight_mean': float(np.mean(weight_copy)),
                                    'weight_std': float(np.std(weight_copy))
                                })
                        return hook
                    
                    for name, module in model.named_modules():
                        if isinstance(module, (nn.Linear, nn.Conv2d)):
                            module.register_forward_hook(make_hook(name))
            
            print("PyTorch hook system demonstrated (would need actual model)")
            
        except ImportError:
            print("PyTorch not available")
    
    def monitor_ollama_internals(self, model_name: str = "deepseek-coder:1.3b"):
        """
        Monitor Ollama's internal state during model execution
        """
        print(f"\n=== Monitoring {model_name} Runtime Weights ===")
        
        import ollama
        
        # Load model
        print("Loading model...")
        ollama.generate(model=model_name, prompt="hello", keep_alive="24h")
        
        # Get process info
        processes = self.get_gpu_memory_info()
        if not processes:
            print("No Ollama GPU processes found!")
            return
        
        ollama_process = processes[0]
        print(f"Found Ollama process: PID={ollama_process['pid']}, Memory={ollama_process['memory_mb']}MB")
        
        # Approach 1: Try memory dump (requires sudo)
        print("\n1. Attempting memory dump approach...")
        memory_dump = self.dump_gpu_memory_via_gdb(ollama_process['pid'], '/tmp/gpu_dump.bin')
        
        # Approach 2: CUDA tools
        self.use_cuda_memcpy()
        
        # Approach 3: PyTorch hooks (demonstration)
        self.extract_via_pytorch_hooks()
        
        # Approach 4: Monitor via repeated inference
        print("\n4. Monitoring via behavioral analysis...")
        
        # Baseline behavior
        baseline_responses = []
        baseline_embeddings = []
        
        test_prompt = "def recursive_function(n):"
        
        print("Getting baseline...")
        for i in range(5):
            # Get response
            resp = ollama.generate(
                model=model_name,
                prompt=test_prompt,
                options={"temperature": 0.0, "seed": 42, "num_predict": 20},
                keep_alive="24h"
            )
            baseline_responses.append(resp['response'])
            
            # Get embedding
            emb = ollama.embeddings(model=model_name, prompt=test_prompt)
            baseline_embeddings.append(emb['embedding'][:10])  # First 10 dims
        
        # Heavy usage phase
        print("\nHeavy usage phase (100 iterations)...")
        for i in range(100):
            ollama.generate(
                model=model_name,
                prompt=f"Write a recursive function for problem {i}",
                options={"temperature": 0.5},
                keep_alive="24h"
            )
            if i % 20 == 0:
                print(f"  Iteration {i}/100")
        
        # Check for changes
        print("\nChecking for behavioral changes...")
        post_responses = []
        post_embeddings = []
        
        for i in range(5):
            resp = ollama.generate(
                model=model_name,
                prompt=test_prompt,
                options={"temperature": 0.0, "seed": 42, "num_predict": 20},
                keep_alive="24h"
            )
            post_responses.append(resp['response'])
            
            emb = ollama.embeddings(model=model_name, prompt=test_prompt)
            post_embeddings.append(emb['embedding'][:10])
        
        # Analysis
        response_changed = baseline_responses != post_responses
        
        embedding_drift = 0
        for base, post in zip(baseline_embeddings, post_embeddings):
            drift = np.linalg.norm(np.array(base) - np.array(post))
            embedding_drift = max(embedding_drift, drift)
        
        print("\n=== RESULTS ===")
        print(f"Response consistency: {'CHANGED' if response_changed else 'UNCHANGED'}")
        print(f"Max embedding drift: {embedding_drift:.8f}")
        
        if response_changed or embedding_drift > 0.0001:
            print("\n⚠️  BEHAVIORAL CHANGES DETECTED!")
            print("This suggests possible runtime weight modifications!")
        else:
            print("\n✓ No behavioral changes detected")
            print("Weights appear static during inference")
        
        # Save results
        self.results['measurements'].append({
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'response_changed': response_changed,
            'embedding_drift': float(embedding_drift),
            'baseline_responses': baseline_responses[:2],  # Sample
            'post_responses': post_responses[:2]  # Sample
        })
        
        with open('gpu_weight_extraction_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\nResults saved to gpu_weight_extraction_results.json")

if __name__ == "__main__":
    extractor = GPUMemoryExtractor()
    
    print("GPU Runtime Weight Extraction Experiment")
    print("="*50)
    print("\nNOTE: Direct GPU memory extraction requires:")
    print("- CUDA toolkit for memory copy")
    print("- sudo access for memory dumps")
    print("- Or integration with Ollama internals")
    print("\nUsing behavioral analysis as proxy...")
    
    extractor.monitor_ollama_internals()