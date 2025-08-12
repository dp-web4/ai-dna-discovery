#!/usr/bin/env python3
"""
Monitor the progress of the language evolution experiment
"""

import json
import glob
import os
from datetime import datetime

def check_progress():
    """Check current progress of the experiment"""
    results_dir = "common_language_results"
    
    # Find latest checkpoint
    checkpoints = glob.glob(os.path.join(results_dir, "checkpoint_round_*.json"))
    if not checkpoints:
        print("No checkpoints found yet. Experiment may not have started.")
        return
    
    # Sort by round number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest = checkpoints[-1]
    
    print(f"Latest checkpoint: {os.path.basename(latest)}")
    
    with open(latest, 'r') as f:
        data = json.load(f)
    
    print(f"\nRound: {data['round']}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Consensus vocabulary size: {data['consensus_size']}")
    print(f"Total patterns in consensus: {len(data['consensus_vocabulary'])}")
    
    # Show some consensus patterns
    if data['consensus_vocabulary']:
        print(f"\nSample consensus patterns:")
        for pattern in data['consensus_vocabulary'][:10]:
            print(f"  - {pattern}")
    
    # Show model vocabulary sizes
    print(f"\nModel vocabulary sizes:")
    for model, vocab in data['model_vocabularies'].items():
        print(f"  {model}: {len(vocab)} patterns")
    
    # Estimate completion
    if data['round'] < 100:
        print(f"\nProgress: {data['round']}/100 rounds ({data['round']}%)")
        # Rough time estimate
        checkpoint_time = datetime.fromisoformat(data['timestamp'])
        elapsed = datetime.now() - checkpoint_time
        print(f"Time since checkpoint: {elapsed}")

if __name__ == "__main__":
    print("="*60)
    print("LANGUAGE EVOLUTION PROGRESS CHECK")
    print("="*60)
    check_progress()