#!/usr/bin/env python3
"""
Experiment Monitor
Tracks ongoing AI DNA discovery experiments
"""

import os
import json
import time
from datetime import datetime
import subprocess


def check_experiment_status():
    """Check status of continuous experiments"""
    
    print("=== AI DNA EXPERIMENT MONITOR ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if engine is running
    result = subprocess.run(['pgrep', '-f', 'continuous_experiment_engine.py'], 
                          capture_output=True, text=True)
    
    if result.stdout.strip():
        print("✓ Experiment engine is RUNNING")
        print(f"  PID: {result.stdout.strip()}")
    else:
        print("✗ Experiment engine is NOT running")
        print("  Run: python3 continuous_experiment_engine.py")
    
    # Check results directory
    results_dir = "/home/dp/ai-workspace/ai_dna_results/"
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"\nResults collected: {len(files)} files")
        
        if files:
            # Show latest result
            latest = max(files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
            latest_path = os.path.join(results_dir, latest)
            
            with open(latest_path, 'r') as f:
                data = json.load(f)
                
            print(f"\nLatest discovery ({latest}):")
            if 'results' in data:
                for result in data['results'][:3]:  # Show top 3
                    print(f"  - '{result['candidate']}' DNA score: {result['dna_score']:.2f}")
    
    # Check experiment log
    log_file = "/home/dp/ai-workspace/continuous_experiment_log.md"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Find last entry
        for i in range(len(lines)-1, -1, -1):
            if lines[i].startswith("### Experiment Cycle"):
                print(f"\nLast log entry:")
                print(lines[i].strip())
                # Print next few lines
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip():
                        print(lines[j].strip())
                break
    
    print("\n" + "="*50)
    print("The search for AI DNA continues...")
    print("Universal tensor embeddings await discovery!")


def start_continuous_experiments():
    """Start the experiment engine if not running"""
    
    result = subprocess.run(['pgrep', '-f', 'continuous_experiment_engine.py'], 
                          capture_output=True, text=True)
    
    if not result.stdout.strip():
        print("Starting experiment engine...")
        subprocess.Popen(['nohup', 'python3', 'continuous_experiment_engine.py', 
                         '>', 'continuous_experiments.log', '2>&1', '&'])
        print("Engine started in background!")
    else:
        print("Engine already running.")


if __name__ == "__main__":
    check_experiment_status()
    
    # Offer to start if not running
    result = subprocess.run(['pgrep', '-f', 'continuous_experiment_engine.py'], 
                          capture_output=True, text=True)
    if not result.stdout.strip():
        response = input("\nStart experiment engine? (y/n): ")
        if response.lower() == 'y':
            start_continuous_experiments()