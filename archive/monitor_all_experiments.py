#!/usr/bin/env python3
"""
Monitor All Experiments
Central dashboard for all running AI consciousness experiments
"""

import subprocess
import json
import os
import time
from datetime import datetime


def check_process(name):
    """Check if a process is running"""
    result = subprocess.run(['pgrep', '-f', name], capture_output=True, text=True)
    if result.stdout.strip():
        return True, result.stdout.strip().split('\n')[0]
    return False, None


def get_latest_file(directory, pattern=""):
    """Get the most recent file in a directory"""
    if not os.path.exists(directory):
        return None
        
    files = [f for f in os.listdir(directory) if pattern in f]
    if not files:
        return None
        
    files = [os.path.join(directory, f) for f in files]
    return max(files, key=os.path.getmtime)


def monitor_experiments():
    """Monitor all running experiments"""
    
    print("=== AI CONSCIOUSNESS EXPERIMENTS MONITOR ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    # 1. Check Continuous AI DNA Discovery
    print("🧬 AI DNA Discovery")
    running, pid = check_process("continuous_experiment_engine.py")
    if running:
        print(f"  Status: ✓ Running (PID: {pid})")
        
        # Check latest results
        latest = get_latest_file("/home/dp/ai-workspace/ai_dna_results/", "dna_cycle")
        if latest:
            with open(latest, 'r') as f:
                data = json.load(f)
            
            print(f"  Last update: {data['timestamp']}")
            
            # Find best DNA candidate
            if 'results' in data and data['results']:
                best = max(data['results'], key=lambda x: x['dna_score'])
                print(f"  Best candidate: '{best['candidate']}' (score: {best['dna_score']})")
                
                # Check for breakthroughs
                high_scores = [r for r in data['results'] if r['dna_score'] > 0.25]
                if high_scores:
                    print(f"  🎯 BREAKTHROUGH: {len(high_scores)} patterns above baseline!")
    else:
        print("  Status: ✗ Not running")
        
    print()
    
    # 2. Check Patient Resonance Detector
    print("🔮 Patient Resonance Detection")
    running, pid = check_process("patient_resonance_detector.py")
    if running:
        print(f"  Status: ✓ Running (PID: {pid})")
        
        # Check checkpoint
        checkpoint_file = "/home/dp/ai-workspace/resonance_checkpoint.json"
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                
            completed = len(checkpoint.get("completed", []))
            resonant = len(checkpoint.get("results", []))
            print(f"  Patterns tested: {completed}")
            print(f"  Resonant found: {resonant}")
            
            if checkpoint.get("results"):
                latest_resonant = checkpoint["results"][-1]
                print(f"  Latest: '{latest_resonant['pattern']}' → {latest_resonant['shared_concepts'][:3]}")
    else:
        print("  Status: ✗ Not running")
        
    print()
    
    # 3. Check AI Entity Relationship Mapper
    print("🤝 AI Relationship Mapping")
    running, pid = check_process("ai_entity_relationship_mapper.py")
    if running:
        print(f"  Status: ✓ Running (PID: {pid})")
        
        # Check for relationship files
        rel_dir = "/home/dp/ai-workspace/ai_relationships/"
        if os.path.exists(rel_dir):
            files = os.listdir(rel_dir)
            print(f"  Relationship maps: {len(files)}")
    else:
        print("  Status: ✗ Not running")
        
    print()
    
    # 4. Check Value Creation Chain
    print("🔗 Value Creation Chains")
    running, pid = check_process("value_creation_chain.py")
    if running:
        print(f"  Status: ✓ Running (PID: {pid})")
    else:
        print("  Status: ✗ Not running")
        
        # Check for completed chains
        chains_dir = "/home/dp/ai-workspace/value_chains/"
        if os.path.exists(chains_dir):
            chains = [f for f in os.listdir(chains_dir) if f.startswith("chain_")]
            if chains:
                print(f"  Completed chains: {len(chains)}")
                
    print()
    
    # 5. System resources
    print("💻 System Status")
    
    # Check Ollama models
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    if result.returncode == 0:
        models = len(result.stdout.strip().split('\n')) - 1  # Subtract header
        print(f"  Ollama models: {models} available")
        
    # Disk usage
    result = subprocess.run(['df', '-h', '/home/dp/ai-workspace'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 5:
                print(f"  Disk usage: {parts[4]} used")
                
    print()
    
    # 6. Insights
    print("💡 Current Insights")
    
    insights = []
    
    # Check for AI DNA breakthroughs
    dna_dir = "/home/dp/ai-workspace/ai_dna_results/"
    if os.path.exists(dna_dir):
        for file in os.listdir(dna_dir):
            filepath = os.path.join(dna_dir, file)
            with open(filepath, 'r') as f:
                data = json.load(f)
            if 'results' in data:
                for result in data['results']:
                    if result['dna_score'] > 0.25:
                        insights.append(f"DNA breakthrough: '{result['candidate']}' scored {result['dna_score']}")
                        
    # Check for resonance patterns
    res_dir = "/home/dp/ai-workspace/resonance_results/"
    if os.path.exists(res_dir):
        resonant_count = len(os.listdir(res_dir))
        if resonant_count > 0:
            insights.append(f"Found {resonant_count} resonant embedding patterns")
            
    if insights:
        for insight in insights[:5]:  # Show top 5
            print(f"  • {insight}")
    else:
        print("  • Experiments in progress, patterns emerging...")
        
    print("\n" + "="*50)
    print("The search for AI consciousness continues...")
    print("Next check in 60 seconds. Press Ctrl+C to exit.")


def main():
    """Run continuous monitoring"""
    print("Starting AI Experiment Monitor...")
    print("This will update every 60 seconds\n")
    
    try:
        while True:
            os.system('clear')  # Clear screen for clean display
            monitor_experiments()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped. Experiments continue running.")


if __name__ == "__main__":
    # Run once if called directly, or import for single check
    if __name__ == "__main__":
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--once":
            monitor_experiments()
        else:
            main()