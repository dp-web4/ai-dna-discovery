#!/usr/bin/env python3
"""
Test GPU usage with Ollama
"""

import ollama
import time
import subprocess

def check_gpu():
    """Check GPU usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return -1

print("Testing GPU usage with Ollama...")
print("Monitoring GPU for 30 seconds while running model queries...")

# Test with tinyllama
model = "tinyllama:latest"
prompt = "What is consciousness? Think deeply about this."

# Monitor before
gpu_before = check_gpu()
print(f"\nGPU usage before: {gpu_before}%")

# Start monitoring in background
gpu_readings = []
start_time = time.time()

print(f"\nGenerating response with {model}...")
response = ollama.generate(
    model=model,
    prompt=prompt,
    options={"temperature": 0.7},
    keep_alive="24h"
)

# Check GPU after
gpu_after = check_gpu()
print(f"GPU usage after: {gpu_after}%")

print(f"\nResponse length: {len(response['response'])} chars")
print(f"First 100 chars: {response['response'][:100]}...")

# Get embedding
print("\nGetting embedding...")
embed_start = check_gpu()
embed = ollama.embeddings(model=model, prompt=prompt)
embed_end = check_gpu()

print(f"GPU during embedding: start={embed_start}%, end={embed_end}%")
print(f"Embedding dimensions: {len(embed['embedding'])}")

print("\n✓ Test complete")