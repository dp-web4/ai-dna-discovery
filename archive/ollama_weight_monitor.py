#!/usr/bin/env python3
"""
Monitor Ollama model weights for runtime changes
Tests if weights remain static or show any drift
"""

import os
import json
import time
import hashlib
import subprocess
from datetime import datetime
import numpy as np
from pathlib import Path

class OllamaWeightMonitor:
    def __init__(self):
        # Find Ollama models directory
        self.ollama_dir = self.find_ollama_models_dir()
        self.results = {
            "theory": "Testing if model weights change during runtime",
            "measurements": []
        }
        
    def find_ollama_models_dir(self):
        """Find where Ollama stores model files"""
        possible_paths = [
            os.path.expanduser("~/.ollama/models"),
            "/usr/share/ollama/models",
            "/var/lib/ollama/models",
            os.path.expanduser("~/ollama/models")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found Ollama models at: {path}")
                return path
                
        # Try to find it via ollama info
        try:
            result = subprocess.run(["find", "/", "-name", "*.gguf", "-path", "*ollama*", "2>/dev/null"], 
                                  capture_output=True, text=True, shell=True)
            if result.stdout:
                model_path = result.stdout.strip().split('\n')[0]
                models_dir = os.path.dirname(os.path.dirname(model_path))
                print(f"Found Ollama models via search: {models_dir}")
                return models_dir
        except:
            pass
            
        print("Warning: Could not find Ollama models directory")
        return None
    
    def get_model_file_info(self, model_name: str):
        """Get information about a model's GGUF file"""
        if not self.ollama_dir:
            return None
            
        # Search for the model file
        model_name_clean = model_name.replace(":", "-")
        
        for root, dirs, files in os.walk(self.ollama_dir):
            for file in files:
                if file.endswith('.gguf') and model_name_clean in root:
                    full_path = os.path.join(root, file)
                    stat = os.stat(full_path)
                    
                    # Get file hash (first 1MB for speed)
                    hash_md5 = hashlib.md5()
                    with open(full_path, 'rb') as f:
                        chunk = f.read(1024*1024)  # First 1MB
                        hash_md5.update(chunk)
                    
                    return {
                        'path': full_path,
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'hash_1mb': hash_md5.hexdigest(),
                        'found': True
                    }
        
        return {'found': False, 'model': model_name}
    
    def compute_weight_checksum(self, model_path: str, sample_size: int = 10):
        """
        Compute checksums at various points in the model file
        This helps detect if ANY part of the weights change
        """
        checksums = {}
        file_size = os.path.getsize(model_path)
        
        # Sample at different positions
        positions = np.linspace(0, file_size - 1024*1024, sample_size, dtype=int)
        
        with open(model_path, 'rb') as f:
            for i, pos in enumerate(positions):
                f.seek(pos)
                chunk = f.read(1024*1024)  # 1MB chunks
                checksums[f'position_{i}'] = hashlib.sha256(chunk).hexdigest()[:16]
        
        return checksums
    
    def monitor_model_during_use(self, model_name: str = "deepseek-coder:1.3b", duration_minutes: int = 5):
        """
        Monitor a model's weights while actively using it
        """
        print(f"\n=== Monitoring {model_name} for {duration_minutes} minutes ===")
        
        # Get initial model info
        initial_info = self.get_model_file_info(model_name)
        if not initial_info or not initial_info['found']:
            print(f"Error: Could not find model file for {model_name}")
            print("Trying alternate approach...")
            return self.monitor_via_embeddings(model_name, duration_minutes)
        
        print(f"Model file: {initial_info['path']}")
        print(f"Size: {initial_info['size'] / 1024**3:.2f} GB")
        
        # Get initial checksums
        initial_checksums = self.compute_weight_checksum(initial_info['path'])
        
        # Load model into memory
        import ollama
        print("\nLoading model into GPU...")
        ollama.generate(model=model_name, prompt="", keep_alive="24h")
        
        # Monitor loop
        start_time = time.time()
        measurement_count = 0
        
        while (time.time() - start_time) < (duration_minutes * 60):
            measurement_count += 1
            elapsed = (time.time() - start_time) / 60
            
            print(f"\n[{elapsed:.1f} min] Measurement #{measurement_count}")
            
            # Use the model actively
            prompts = [
                "def factorial(n): return",
                "The meaning of life is",
                "while True: print(",
                "import numpy as np\n",
                "class NeuralNetwork:\n    def __init__(self):"
            ]
            
            for prompt in prompts:
                ollama.generate(
                    model=model_name, 
                    prompt=prompt,
                    options={"temperature": 0.0, "num_predict": 20},
                    keep_alive="24h"
                )
            
            # Check file info
            current_info = self.get_model_file_info(model_name)
            
            # Check for changes
            changes_detected = []
            
            if current_info['size'] != initial_info['size']:
                changes_detected.append(f"SIZE CHANGED: {initial_info['size']} -> {current_info['size']}")
            
            if current_info['mtime'] != initial_info['mtime']:
                changes_detected.append(f"MTIME CHANGED: {initial_info['mtime']} -> {current_info['mtime']}")
            
            if current_info['hash_1mb'] != initial_info['hash_1mb']:
                changes_detected.append(f"HASH CHANGED: {initial_info['hash_1mb']} -> {current_info['hash_1mb']}")
            
            # Deep checksum comparison
            current_checksums = self.compute_weight_checksum(current_info['path'])
            checksum_changes = 0
            for pos, checksum in current_checksums.items():
                if checksum != initial_checksums.get(pos):
                    checksum_changes += 1
            
            if checksum_changes > 0:
                changes_detected.append(f"CHECKSUM CHANGES: {checksum_changes}/{len(current_checksums)} positions")
            
            # Record measurement
            measurement = {
                'timestamp': datetime.now().isoformat(),
                'elapsed_minutes': elapsed,
                'measurement_num': measurement_count,
                'changes_detected': changes_detected,
                'file_stats': current_info
            }
            
            self.results['measurements'].append(measurement)
            
            if changes_detected:
                print(f"⚠️  CHANGES DETECTED: {changes_detected}")
            else:
                print("✓ No changes detected")
            
            # Wait before next measurement
            time.sleep(30)
        
        # Final analysis
        self.analyze_results()
    
    def monitor_via_embeddings(self, model_name: str, duration_minutes: int):
        """
        Alternative monitoring via embedding stability
        """
        print("\nUsing embedding-based monitoring...")
        import ollama
        
        # Reference prompts
        test_prompts = [
            "The sky is",
            "def main():",
            "1 + 1 =",
            "Hello world"
        ]
        
        # Get baseline embeddings
        print("Getting baseline embeddings...")
        baseline_embeddings = {}
        for prompt in test_prompts:
            resp = ollama.embeddings(model=model_name, prompt=prompt)
            baseline_embeddings[prompt] = np.array(resp['embedding'])
        
        # Monitor loop
        start_time = time.time()
        measurement_count = 0
        
        while (time.time() - start_time) < (duration_minutes * 60):
            measurement_count += 1
            elapsed = (time.time() - start_time) / 60
            
            print(f"\n[{elapsed:.1f} min] Measurement #{measurement_count}")
            
            # Heavy model usage
            for _ in range(10):
                ollama.generate(
                    model=model_name,
                    prompt="Write a recursive function to calculate fibonacci numbers",
                    options={"temperature": 0.7, "num_predict": 50},
                    keep_alive="24h"
                )
            
            # Check embedding stability
            max_drift = 0
            for prompt in test_prompts:
                current_resp = ollama.embeddings(model=model_name, prompt=prompt)
                current_embedding = np.array(current_resp['embedding'])
                
                # Calculate drift
                drift = np.linalg.norm(current_embedding - baseline_embeddings[prompt])
                max_drift = max(max_drift, drift)
                
                if drift > 0.001:  # Threshold for significant change
                    print(f"  Embedding drift for '{prompt}': {drift:.6f}")
            
            measurement = {
                'timestamp': datetime.now().isoformat(),
                'elapsed_minutes': elapsed,
                'max_embedding_drift': float(max_drift),
                'significant_drift': max_drift > 0.001
            }
            
            self.results['measurements'].append(measurement)
            
            if max_drift > 0.001:
                print(f"⚠️  EMBEDDING DRIFT DETECTED: {max_drift:.6f}")
            else:
                print("✓ Embeddings stable")
            
            time.sleep(30)
        
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze monitoring results"""
        print("\n=== ANALYSIS ===")
        
        total_measurements = len(self.results['measurements'])
        changes_found = sum(1 for m in self.results['measurements'] 
                          if m.get('changes_detected') or m.get('significant_drift'))
        
        print(f"Total measurements: {total_measurements}")
        print(f"Measurements with changes: {changes_found}")
        
        if changes_found > 0:
            print("\n⚠️  RUNTIME PLASTICITY DETECTED!")
            print("The model weights or behavior changed during use!")
        else:
            print("\n✓ NO RUNTIME PLASTICITY DETECTED")
            print("Model weights remained static during inference")
        
        # Save results
        output_file = f"weight_monitoring_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    monitor = OllamaWeightMonitor()
    
    # First, let's find where models are stored
    print("Searching for Ollama model files...")
    
    # Test with deepseek-coder as per your intuition
    monitor.monitor_model_during_use(
        model_name="deepseek-coder:1.3b",
        duration_minutes=5  # Start with 5 minutes
    )