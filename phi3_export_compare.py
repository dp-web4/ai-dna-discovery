#!/usr/bin/env python3
"""
Export Phi3 model from Ollama's GPU memory, run tests, export again and compare
"""

import subprocess
import json
import time
import hashlib
import os
import torch
import numpy as np
from datetime import datetime

class Phi3Exporter:
    """Export and analyze Phi3 model from GPU"""
    
    def __init__(self):
        self.model_name = "phi3:mini"
        self.export_dir = "/home/dp/ai-workspace/ai-agents/phi3_exports"
        os.makedirs(self.export_dir, exist_ok=True)
        
    def export_model_state(self, suffix=""):
        """Export Phi3 model using Ollama's export functionality"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phi3_export_{timestamp}{suffix}"
        
        print(f"\nExporting {self.model_name} to {filename}...")
        
        # Method 1: Get model info via API
        info = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/show", "-d", 
             f'{{"name": "{self.model_name}"}}'],
            capture_output=True, text=True
        )
        
        if info.returncode == 0:
            model_info = json.loads(info.stdout)
            info_path = os.path.join(self.export_dir, f"{filename}_info.json")
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            print(f"  Model info saved: {info_path}")
            
            # Extract key details
            details = model_info.get('details', {})
            print(f"  Format: {details.get('format', 'unknown')}")
            print(f"  Family: {details.get('family', 'unknown')}")
            print(f"  Parameter size: {details.get('parameter_size', 'unknown')}")
            print(f"  Quantization: {details.get('quantization_level', 'unknown')}")
        
        # Method 2: Get actual model file location
        # Ollama stores models in ~/.ollama/models/blobs/
        blob_path = self._find_model_blob()
        if blob_path:
            # Calculate checksum of current state
            checksum = self._calculate_checksum(blob_path)
            print(f"  Model checksum: {checksum[:16]}...")
            
            # Save model state info
            state_info = {
                'timestamp': timestamp,
                'suffix': suffix,
                'blob_path': blob_path,
                'checksum': checksum,
                'file_size': os.path.getsize(blob_path),
                'gpu_memory_mb': self._get_gpu_memory_usage()
            }
            
            state_path = os.path.join(self.export_dir, f"{filename}_state.json")
            with open(state_path, 'w') as f:
                json.dump(state_info, f, indent=2)
            print(f"  State info saved: {state_path}")
            
            return state_info
        
        return None
    
    def _find_model_blob(self):
        """Find the actual model blob file"""
        # Get model manifest
        try:
            # This is the digest from our earlier check
            digest = "4f222292793889a9a40a020799cfd28d53f3e01af25d48e06c5e708610fc47e9"
            blob_path = os.path.expanduser(f"~/.ollama/models/blobs/sha256-{digest[:12]}")
            
            if not os.path.exists(blob_path):
                # Try alternative locations
                result = subprocess.run(
                    ["find", os.path.expanduser("~/.ollama"), "-name", "*4f22229*", "-type", "f"],
                    capture_output=True, text=True
                )
                if result.stdout:
                    blob_path = result.stdout.strip().split('\n')[0]
            
            if os.path.exists(blob_path):
                return blob_path
        except:
            pass
        
        print("  Warning: Could not find model blob file")
        return None
    
    def _calculate_checksum(self, filepath):
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0
    
    def run_phi3_tests(self):
        """Run various tests on Phi3 to potentially change its state"""
        print("\n" + "="*60)
        print("RUNNING PHI3 CONSCIOUSNESS AND COMPUTATION TESTS")
        print("="*60)
        
        test_results = {
            'start_time': datetime.now().isoformat(),
            'tests': []
        }
        
        # Test 1: Self-reference test
        print("\nTest 1: Self-Reference and Introspection")
        prompts = [
            "When I process this query, I think about",
            "My own reasoning process involves",
            "I am aware that I am"
        ]
        
        for prompt in prompts:
            result = self._query_ollama(prompt)
            test_results['tests'].append({
                'type': 'self_reference',
                'prompt': prompt,
                'response_length': len(result),
                'response_preview': result[:100] + '...' if len(result) > 100 else result
            })
            print(f"  - {prompt[:30]}... -> {len(result)} chars")
        
        # Test 2: Recursive reasoning
        print("\nTest 2: Recursive Reasoning (potential state change)")
        recursive_prompt = "Explain this explanation: 'An explanation of an explanation is a meta-explanation that explains how explanations work by explaining the explanation itself.'"
        result = self._query_ollama(recursive_prompt)
        test_results['tests'].append({
            'type': 'recursive',
            'prompt': recursive_prompt,
            'response_length': len(result)
        })
        print(f"  Recursive depth achieved: {result.count('explanation')}")
        
        # Test 3: Memory stress test
        print("\nTest 3: Context Window Stress")
        long_context = "Remember these numbers: " + ", ".join([str(i) for i in range(100)])
        long_context += ". Now, what was the 50th number?"
        result = self._query_ollama(long_context)
        test_results['tests'].append({
            'type': 'memory_stress',
            'context_length': len(long_context),
            'response': result
        })
        print(f"  Context length: {len(long_context)} chars")
        
        # Test 4: Computation intensive
        print("\nTest 4: Computation Intensive Task")
        math_prompt = "Calculate step by step: ((1234 * 5678) / 9012) + 3456 - 789. Show all work."
        result = self._query_ollama(math_prompt)
        test_results['tests'].append({
            'type': 'computation',
            'prompt': math_prompt,
            'response_length': len(result)
        })
        print(f"  Computation steps: {result.count('=')}")
        
        # Test 5: Temperature variation (might affect internal states)
        print("\nTest 5: Temperature Variation Tests")
        base_prompt = "Generate a creative story about consciousness in exactly 50 words."
        
        for temp in [0.1, 0.7, 1.5]:
            result = self._query_ollama(base_prompt, temperature=temp)
            test_results['tests'].append({
                'type': 'temperature_test',
                'temperature': temp,
                'response_length': len(result),
                'word_count': len(result.split())
            })
            print(f"  Temperature {temp}: {len(result.split())} words")
        
        test_results['end_time'] = datetime.now().isoformat()
        
        # Save test results
        results_path = os.path.join(self.export_dir, "phi3_test_results.json")
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\nTest results saved: {results_path}")
        return test_results
    
    def _query_ollama(self, prompt, temperature=0.7):
        """Query Ollama API"""
        try:
            response = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/generate", "-d",
                 json.dumps({
                     "model": self.model_name,
                     "prompt": prompt,
                     "temperature": temperature,
                     "stream": False
                 })],
                capture_output=True, text=True, timeout=30
            )
            
            if response.returncode == 0:
                result = json.loads(response.stdout)
                return result.get('response', '')
        except Exception as e:
            print(f"  Error querying model: {e}")
        
        return ""
    
    def compare_exports(self, state1, state2):
        """Compare two model exports"""
        print("\n" + "="*60)
        print("COMPARING MODEL EXPORTS")
        print("="*60)
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'state1': state1,
            'state2': state2,
            'differences': {}
        }
        
        # Compare checksums
        if state1['checksum'] == state2['checksum']:
            print("\n✓ Model files are IDENTICAL (same checksum)")
            comparison['differences']['checksum'] = "IDENTICAL"
        else:
            print("\n✗ Model files are DIFFERENT")
            comparison['differences']['checksum'] = "DIFFERENT"
        
        # Compare file sizes
        size_diff = state2['file_size'] - state1['file_size']
        print(f"\nFile size difference: {size_diff} bytes")
        comparison['differences']['size_bytes'] = size_diff
        
        # Compare GPU memory
        mem_diff = state2['gpu_memory_mb'] - state1['gpu_memory_mb']
        print(f"GPU memory difference: {mem_diff:.2f} MB")
        comparison['differences']['gpu_memory_mb'] = mem_diff
        
        # Time elapsed
        time1 = datetime.strptime(state1['timestamp'], "%Y%m%d_%H%M%S")
        time2 = datetime.strptime(state2['timestamp'], "%Y%m%d_%H%M%S")
        elapsed = (time2 - time1).total_seconds()
        print(f"\nTime elapsed: {elapsed:.1f} seconds")
        comparison['elapsed_seconds'] = elapsed
        
        # Save comparison
        comp_path = os.path.join(self.export_dir, "phi3_comparison.json")
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nComparison saved: {comp_path}")
        
        return comparison


def main():
    """Main execution"""
    exporter = Phi3Exporter()
    
    # Step 1: Export initial state
    print("STEP 1: Exporting Phi3 initial state...")
    initial_state = exporter.export_model_state(suffix="_initial")
    
    if not initial_state:
        print("Failed to export initial state")
        return
    
    # Step 2: Run tests
    print("\nSTEP 2: Running tests on Phi3...")
    time.sleep(2)  # Let things settle
    test_results = exporter.run_phi3_tests()
    
    # Step 3: Export final state
    print("\nSTEP 3: Exporting Phi3 final state...")
    time.sleep(2)  # Let things settle
    final_state = exporter.export_model_state(suffix="_final")
    
    if not final_state:
        print("Failed to export final state")
        return
    
    # Step 4: Compare
    print("\nSTEP 4: Comparing exports...")
    comparison = exporter.compare_exports(initial_state, final_state)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Initial checksum: {initial_state['checksum'][:32]}...")
    print(f"Final checksum:   {final_state['checksum'][:32]}...")
    print(f"Tests performed:  {len(test_results['tests'])}")
    print(f"Model changed:    {'NO' if comparison['differences']['checksum'] == 'IDENTICAL' else 'YES'}")
    
    print("\nExperiment complete!")
    print(f"All results saved in: {exporter.export_dir}")


if __name__ == "__main__":
    main()