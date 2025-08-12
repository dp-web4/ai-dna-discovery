#!/usr/bin/env python3
"""
Direct GPU test with real-time monitoring
"""

import subprocess
import time
import threading
import ollama

def monitor_gpu():
    """Monitor GPU usage in real-time"""
    print("\nGPU Usage (updating every 0.5s, Ctrl+C to stop):")
    print("-" * 40)
    
    for i in range(20):  # 10 seconds
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            gpu_util, mem_used = result.stdout.strip().split(', ')
            print(f"\r[{i*0.5:.1f}s] GPU: {gpu_util}%, Memory: {mem_used} MB", end='', flush=True)
            time.sleep(0.5)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")
            break
    print("\n")

# Start monitoring in background
monitor_thread = threading.Thread(target=monitor_gpu)
monitor_thread.start()

# Give monitoring a moment to start
time.sleep(0.5)

# Generate with ollama
print("\nGenerating with tinyllama...")
response = ollama.generate(
    model="tinyllama:latest",
    prompt="Count from 1 to 100 slowly, thinking about each number.",
    options={"temperature": 0.5, "num_predict": 500}
)

# Wait for monitoring to finish
monitor_thread.join()

print(f"\nGenerated {len(response['response'])} characters")
print("First 100 chars:", response['response'][:100])