#!/usr/bin/env python3
"""
Quick test to monitor GPU during orchestra operations
"""

import ollama
import time
import subprocess
import threading

def monitor_gpu(duration=10):
    """Monitor GPU for duration seconds"""
    print("GPU Monitoring:")
    for i in range(duration * 2):  # Check every 0.5s
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        util = result.stdout.strip()
        print(f"\r[{i*0.5:.1f}s] GPU: {util}%", end='', flush=True)
        time.sleep(0.5)
    print()

# Start monitoring
monitor_thread = threading.Thread(target=monitor_gpu, args=(15,))
monitor_thread.start()

# Simulate orchestra interaction
print("\nRunning orchestra simulation...")
models = ["phi3:mini", "gemma:2b", "tinyllama:latest"]

for round in range(3):
    print(f"\nRound {round + 1}:")
    for model in models:
        prompt = f"Continue this story in 50 words: The quantum particles danced..."
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.7, "num_predict": 50}
        )
        print(f"  {model}: {len(response['response'])} chars generated")

monitor_thread.join()
print("\nTest complete!")