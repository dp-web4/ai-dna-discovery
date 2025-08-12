#!/usr/bin/env python3
"""
Run consciousness probe with GPU monitoring
"""

import time
import subprocess
import threading
from phase_1_consciousness_field import ConsciousnessProbeExperiment

class GPUMonitor:
    def __init__(self):
        self.running = True
        self.max_usage = 0
        self.readings = []
        
    def check_gpu(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            return int(result.stdout.strip())
        except:
            return -1
            
    def monitor(self):
        while self.running:
            usage = self.check_gpu()
            self.readings.append(usage)
            if usage > self.max_usage:
                self.max_usage = usage
            time.sleep(0.5)
            
    def start(self):
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()
        
    def stop(self):
        self.running = False
        self.thread.join()
        
    def report(self):
        if self.readings:
            avg = sum(self.readings) / len(self.readings)
            print(f"\nGPU Usage Report:")
            print(f"  Max: {self.max_usage}%")
            print(f"  Average: {avg:.1f}%")
            print(f"  Readings: {len(self.readings)}")
            print(f"  Non-zero readings: {sum(1 for r in self.readings if r > 0)}")

print("Running Consciousness Probe with GPU Monitoring")
print("=" * 50)

# Start GPU monitoring
monitor = GPUMonitor()
monitor.start()

try:
    # Run experiment with just one model to keep it focused
    exp = ConsciousnessProbeExperiment()
    exp.models = ["tinyllama:latest"]  # Just one model
    exp.consciousness_prompts = exp.consciousness_prompts[:3]  # Just first 3 prompts
    
    print(f"Testing {len(exp.models)} model(s) with {len(exp.consciousness_prompts)} prompts")
    print("Starting experiment...")
    
    exp.run()
    
    print("\nExperiment completed!")
    
finally:
    # Stop monitoring
    monitor.stop()
    monitor.report()
    
print("\nChecking results...")
results_file = sorted(exp.tracker.phase_dir.glob("consciousness_probe_*.json"))[-1]
print(f"Results saved to: {results_file}")

# Show WiFi activity check
print("\nChecking network activity...")
subprocess.run("ss -tunp 2>/dev/null | grep -E 'ollama|:11434' | head -5", shell=True)