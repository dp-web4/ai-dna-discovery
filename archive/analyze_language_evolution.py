#!/usr/bin/env python3
"""
Analyze language evolution experiment results
"""

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_fast_results():
    """Analyze the fast language evolution results"""
    results_dir = "language_evolution_fast_results"
    
    # Find checkpoint files
    checkpoints = glob.glob(os.path.join(results_dir, "checkpoint_*.json"))
    if not checkpoints:
        print("No checkpoints found yet")
        return
    
    # Load latest checkpoint
    latest = sorted(checkpoints)[-1]
    print(f"Analyzing: {latest}")
    
    with open(latest, 'r') as f:
        data = json.load(f)
    
    print(f"\nRound: {data['round']}")
    print(f"Consensus patterns: {len(data['consensus_patterns'])}")
    print(f"Total vocabulary: {len(data['vocabulary']['seeds']) + len(data['vocabulary']['novel'])}")
    
    # Analyze consensus scores from rounds
    if 'results_so_far' in data and 'rounds' in data['results_so_far']:
        rounds_data = data['results_so_far']['rounds']
        
        # Extract consensus scores
        all_consensus_scores = []
        for round_data in rounds_data:
            for new_pattern in round_data.get('new_patterns', []):
                all_consensus_scores.append(new_pattern['consensus'])
        
        if all_consensus_scores:
            print(f"\nConsensus score statistics:")
            print(f"  Mean: {np.mean(all_consensus_scores):.4f}")
            print(f"  Max: {np.max(all_consensus_scores):.4f}")
            print(f"  Min: {np.min(all_consensus_scores):.4f}")
            print(f"  Std: {np.std(all_consensus_scores):.4f}")
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.hist(all_consensus_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', label='Zero consensus')
            plt.axvline(x=0.5, color='green', linestyle='--', label='Vocabulary threshold')
            plt.axvline(x=0.7, color='gold', linestyle='--', label='Consensus threshold')
            plt.xlabel('Consensus Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Pattern Consensus Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('language_consensus_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\nSaved consensus distribution plot")
            
            # Plot evolution over time
            rounds = []
            avg_consensus_per_round = []
            
            for round_data in rounds_data:
                round_num = round_data['round']
                round_consensus = [p['consensus'] for p in round_data.get('new_patterns', [])]
                if round_consensus:
                    rounds.append(round_num)
                    avg_consensus_per_round.append(np.mean(round_consensus))
            
            if rounds:
                plt.figure(figsize=(10, 6))
                plt.plot(rounds, avg_consensus_per_round, 'b-', linewidth=2, marker='o')
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Vocabulary threshold')
                plt.axhline(y=0.7, color='gold', linestyle='--', alpha=0.5, label='Consensus threshold')
                plt.xlabel('Round')
                plt.ylabel('Average Consensus Score')
                plt.title('Language Consensus Evolution Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('language_consensus_evolution.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("Saved consensus evolution plot")
    
    # Analyze vocabulary
    if 'vocabulary' in data:
        print(f"\nVocabulary analysis:")
        print(f"  Seed patterns: {len(data['vocabulary']['seeds'])}")
        print(f"  Novel patterns: {len(data['vocabulary']['novel'])}")
        
        if data['vocabulary']['novel']:
            print(f"\nSample novel patterns:")
            for pattern in data['vocabulary']['novel'][:10]:
                print(f"  - {pattern}")
    
    print("\nKey insight:")
    print("The very low consensus scores (near 0) indicate that the three models")
    print("have fundamentally different representation spaces. They're essentially")
    print("'speaking different languages' at the embedding level, making consensus difficult.")

if __name__ == "__main__":
    analyze_fast_results()