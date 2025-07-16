#!/usr/bin/env python3
"""Test GPU activity during generation"""

import ollama
import subprocess
import threading
import time

def monitor_gpu():
    """Monitor GPU in separate thread"""
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True)
        util = result.stdout.strip()
        if util and int(util) > 0:
            print(f"GPU: {util}%", end="\r")
        time.sleep(0.1)

# Start GPU monitor
monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
monitor_thread.start()

print("Generating text (watch GPU)...")
response = ollama.generate(
    model="deepseek-coder:1.3b",
    prompt="Write a complete implementation of quicksort in Python with detailed comments",
    options={"num_predict": 500},
    stream=True
)

for chunk in response:
    print(".", end="", flush=True)

print("\nDone!")