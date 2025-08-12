#!/usr/bin/env python3
"""
Quick Memory Test - Check if models are responsive before full experiment
"""

import requests
import time

def quick_test():
    """Quick test to see if models are responsive"""
    models = ["phi3:mini", "tinyllama:latest"]
    pattern = "emerge"
    
    print("=== Quick Memory Test ===")
    print("Checking if models are responsive...\n")
    
    for model in models:
        print(f"Testing {model}...")
        try:
            start = time.time()
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": model, "prompt": pattern},
                timeout=10
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                print(f"  ✓ {model} responded in {elapsed:.2f}s")
            else:
                print(f"  ✗ {model} error: {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ {model} failed: {str(e)}")
    
    print("\nIf models are responsive, run: python3 memory_persistence_experiment.py")

if __name__ == "__main__":
    quick_test()