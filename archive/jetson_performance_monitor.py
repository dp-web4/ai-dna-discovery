#!/usr/bin/env python3
"""
Monitor Jetson performance while running memory tests
Shows real-time GPU, CPU, and memory usage
"""

import json
import time
import urllib.request
import subprocess
import threading
from datetime import datetime

# Global flag for monitoring thread
monitoring = True
performance_data = []

def monitor_jetson():
    """Background thread to collect performance data"""
    global monitoring, performance_data
    
    while monitoring:
        try:
            # Get tegrastats output
            result = subprocess.run(['tegrastats', '--interval', '1000'], 
                                  capture_output=True, text=True, timeout=1)
            if result.stdout:
                stats = result.stdout.strip()
                performance_data.append({
                    'time': datetime.now().isoformat(),
                    'stats': stats
                })
        except:
            pass
        
        # Also get GPU utilization if available
        try:
            nvidia_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                                          '--format=csv,noheader,nounits'], 
                                         capture_output=True, text=True, timeout=1)
            if nvidia_result.returncode == 0 and nvidia_result.stdout:
                gpu_util, mem_used = nvidia_result.stdout.strip().split(', ')
                if performance_data:
                    performance_data[-1]['gpu_util'] = gpu_util
                    performance_data[-1]['gpu_mem_mb'] = mem_used
        except:
            pass
        
        time.sleep(1)

def chat_with_monitoring(model, prompt):
    """Chat with performance monitoring"""
    url = 'http://localhost:11434/api/generate'
    
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    start_time = time.time()
    perf_before = len(performance_data)
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            response_text = result.get('response', 'No response')
    except Exception as e:
        response_text = f"Error: {str(e)}"
    
    elapsed = time.time() - start_time
    perf_after = len(performance_data)
    
    # Get performance during this inference
    inference_perf = performance_data[perf_before:perf_after]
    
    return response_text, elapsed, inference_perf

def run_performance_test():
    """Run memory test with performance monitoring"""
    print("🚀 JETSON ORIN NANO - PERFORMANCE MONITORED MEMORY TEST")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    print(f"Hardware: 40 TOPS, 1024 CUDA cores, 8GB RAM")
    print("=" * 60 + "\n")
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_jetson)
    monitor_thread.daemon = True
    monitor_thread.start()
    print("Performance monitoring started...\n")
    
    # Simple test conversations
    test_prompts = [
        "Hello! I'm Dennis testing the Jetson Orin Nano's AI capabilities.",
        "Previous: User said they are Dennis testing Jetson. Now user asks: What's my name and what am I testing?",
        "Generate a haiku about edge AI and memory."
    ]
    
    context = ""
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {prompt[:60]}...")
        
        full_prompt = context + prompt
        response, elapsed, perf_data = chat_with_monitoring("phi3:mini", full_prompt)
        
        print(f"\nResponse: {response[:150]}{'...' if len(response) > 150 else ''}")
        print(f"Time: {elapsed:.1f}s")
        
        # Analyze performance during inference
        if perf_data:
            print(f"\nPerformance during inference ({len(perf_data)} samples):")
            
            # Parse first and last tegrastats
            if 'stats' in perf_data[0]:
                print(f"Start: {perf_data[0]['stats'][:80]}...")
            if 'stats' in perf_data[-1]:
                print(f"End:   {perf_data[-1]['stats'][:80]}...")
            
            # GPU utilization if available
            gpu_utils = [p.get('gpu_util', 0) for p in perf_data if 'gpu_util' in p]
            if gpu_utils:
                print(f"GPU Utilization: {max(gpu_utils)}% peak")
        
        # Update context
        context = f"Previous: {response[:100]}...\n"
        
        time.sleep(2)  # Brief pause between tests
    
    # Stop monitoring
    global monitoring
    monitoring = False
    time.sleep(2)
    
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"Total performance samples collected: {len(performance_data)}")
    
    # Memory summary
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    mem_gb = int(line.split()[1]) / 1024 / 1024
                    print(f"Final available memory: {mem_gb:.1f} GB")
                    break
    except:
        pass
    
    print("\n✅ Performance monitoring complete!")
    
    # Save performance data
    with open('jetson_performance_log.json', 'w') as f:
        json.dump(performance_data, f, indent=2)
    print("Performance data saved to jetson_performance_log.json")

if __name__ == "__main__":
    run_performance_test()