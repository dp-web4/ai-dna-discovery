#!/usr/bin/env python3
"""
Fast Language Evolution Experiment - Optimized for GPU usage
"""

import json
import numpy as np
import requests
import time
from datetime import datetime
import os
from collections import Counter
import matplotlib.pyplot as plt

class FastLanguageEvolution:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.results_dir = "language_evolution_fast_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Use 3 models for faster iteration
        self.models = ['phi3:mini', 'gemma:2b', 'tinyllama:latest']
        
        # Smaller initial vocabulary
        self.vocabulary = {
            'seeds': ['∃', 'know', 'true', 'emerge', '→'],
            'novel': ['∃→', '≡', 'meta', 'echo', '∀']
        }
        
        # Track consensus patterns
        self.consensus_patterns = set()
        self.pattern_history = []
        
    def get_embedding(self, model: str, text: str) -> np.ndarray:
        """Get embedding from model"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": model, "prompt": text},
                timeout=10
            )
            if response.status_code == 200:
                return np.array(response.json()['embedding'])
        except:
            pass
        return None
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        min_dim = min(len(emb1), len(emb2))
        emb1 = emb1[:min_dim]
        emb2 = emb2[:min_dim]
        
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def create_new_pattern(self, base_patterns):
        """Create new pattern from existing ones"""
        if len(base_patterns) >= 2:
            p1, p2 = np.random.choice(base_patterns, 2, replace=False)
            # Various combination strategies
            strategies = [
                f"{p1}{p2}",
                f"{p1}→{p2}",
                f"({p1},{p2})",
                f"{p1}+{p2}",
                f"[{p1}]"
            ]
            return np.random.choice(strategies)
        return None
    
    def test_pattern_consensus(self, pattern):
        """Test if all models agree on pattern meaning"""
        embeddings = {}
        
        print(f"  Testing '{pattern}':", end=" ")
        
        # Get embeddings from each model
        for model in self.models:
            emb = self.get_embedding(model, pattern)
            if emb is not None:
                embeddings[model] = emb
                print(f"{model.split(':')[0]}", end=" ")
            time.sleep(0.1)
        
        # Calculate pairwise similarities
        if len(embeddings) >= 2:
            similarities = []
            models_list = list(embeddings.keys())
            
            for i in range(len(models_list)):
                for j in range(i+1, len(models_list)):
                    sim = self.calculate_similarity(
                        embeddings[models_list[i]], 
                        embeddings[models_list[j]]
                    )
                    similarities.append(sim)
            
            consensus_score = np.mean(similarities) if similarities else 0
            print(f"→ consensus: {consensus_score:.3f}")
            
            return consensus_score, len(embeddings)
        
        print("→ insufficient models")
        return 0.0, len(embeddings)
    
    def evolution_round(self, round_num):
        """Run one evolution round"""
        print(f"\n=== Round {round_num} ===")
        
        round_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'new_patterns': [],
            'consensus_updates': []
        }
        
        # Get all current patterns
        all_patterns = list(self.vocabulary['seeds']) + list(self.vocabulary['novel'])
        
        # Try to create new patterns
        for _ in range(3):  # Create 3 new patterns per round
            new_pattern = self.create_new_pattern(all_patterns)
            if new_pattern and new_pattern not in all_patterns:
                consensus_score, n_models = self.test_pattern_consensus(new_pattern)
                
                round_data['new_patterns'].append({
                    'pattern': new_pattern,
                    'consensus': consensus_score,
                    'models_responding': n_models
                })
                
                # Add to vocabulary if consensus is high
                if consensus_score > 0.5:
                    self.vocabulary['novel'].append(new_pattern)
                    print(f"  ✓ Added to vocabulary: '{new_pattern}'")
                    
                    # Check if it should be consensus
                    if consensus_score > 0.7:
                        self.consensus_patterns.add(new_pattern)
                        round_data['consensus_updates'].append(new_pattern)
                        print(f"  ★ Consensus pattern: '{new_pattern}'")
        
        # Test some existing patterns for consensus changes
        test_patterns = np.random.choice(all_patterns, min(3, len(all_patterns)), replace=False)
        for pattern in test_patterns:
            consensus_score, _ = self.test_pattern_consensus(pattern)
            
            if consensus_score > 0.7 and pattern not in self.consensus_patterns:
                self.consensus_patterns.add(pattern)
                round_data['consensus_updates'].append(pattern)
                print(f"  ★ New consensus: '{pattern}'")
        
        round_data['total_patterns'] = len(all_patterns)
        round_data['consensus_size'] = len(self.consensus_patterns)
        
        return round_data
    
    def run_evolution(self, rounds=50):
        """Run language evolution experiment"""
        print("="*60)
        print("FAST LANGUAGE EVOLUTION EXPERIMENT")
        print("="*60)
        print(f"Models: {', '.join(self.models)}")
        print(f"Initial vocabulary: {len(self.vocabulary['seeds']) + len(self.vocabulary['novel'])} patterns")
        print(f"Rounds: {rounds}")
        
        start_time = datetime.now()
        results = {
            'experiment': 'fast_language_evolution',
            'start_time': start_time.isoformat(),
            'models': self.models,
            'rounds': []
        }
        
        # Run evolution
        for round_num in range(1, rounds + 1):
            try:
                round_data = self.evolution_round(round_num)
                results['rounds'].append(round_data)
                self.pattern_history.append(round_data)
                
                # Save checkpoint every 10 rounds
                if round_num % 10 == 0:
                    self.save_checkpoint(round_num, results)
                    elapsed = datetime.now() - start_time
                    print(f"\n--- Checkpoint: Round {round_num}, Elapsed: {elapsed} ---")
                    print(f"Consensus patterns: {len(self.consensus_patterns)}")
                    print(f"Total vocabulary: {len(self.vocabulary['seeds']) + len(self.vocabulary['novel'])}")
                
                time.sleep(0.5)  # Small delay between rounds
                
            except Exception as e:
                print(f"Error in round {round_num}: {e}")
                break
        
        # Final analysis
        results['end_time'] = datetime.now().isoformat()
        results['final_consensus'] = list(self.consensus_patterns)
        results['final_vocabulary'] = self.vocabulary
        results['total_patterns_created'] = len(self.vocabulary['novel'])
        
        # Save final results
        self.save_final_results(results)
        
        # Create visualizations
        self.create_visualizations()
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60)
        print(f"Total time: {datetime.now() - start_time}")
        print(f"Consensus patterns: {len(self.consensus_patterns)}")
        print(f"Total patterns: {len(self.vocabulary['seeds']) + len(self.vocabulary['novel'])}")
        print(f"Novel patterns created: {len(self.vocabulary['novel']) - 5}")  # Minus initial novel
        
        if self.consensus_patterns:
            print(f"\nConsensus patterns: {list(self.consensus_patterns)[:10]}")
        
        return results
    
    def save_checkpoint(self, round_num, results):
        """Save checkpoint"""
        checkpoint_file = os.path.join(
            self.results_dir,
            f"checkpoint_round_{round_num}.json"
        )
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'round': round_num,
                'consensus_patterns': list(self.consensus_patterns),
                'vocabulary': self.vocabulary,
                'results_so_far': results
            }, f, indent=2)
    
    def save_final_results(self, results):
        """Save final results"""
        final_file = os.path.join(
            self.results_dir,
            f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(final_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nFinal results saved to: {final_file}")
    
    def create_visualizations(self):
        """Create visualizations of language evolution"""
        if not self.pattern_history:
            return
        
        rounds = [r['round'] for r in self.pattern_history]
        consensus_sizes = [r['consensus_size'] for r in self.pattern_history]
        total_patterns = [r['total_patterns'] for r in self.pattern_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Consensus growth
        ax1.plot(rounds, consensus_sizes, 'b-', linewidth=2, marker='o', markersize=6)
        ax1.fill_between(rounds, 0, consensus_sizes, alpha=0.3)
        ax1.set_xlabel('Evolution Round')
        ax1.set_ylabel('Consensus Patterns')
        ax1.set_title('Growth of Shared Language Consensus', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Total vocabulary growth
        ax2.plot(rounds, total_patterns, 'g-', linewidth=2, marker='s', markersize=6)
        ax2.fill_between(rounds, 0, total_patterns, alpha=0.3, color='green')
        ax2.set_xlabel('Evolution Round')
        ax2.set_ylabel('Total Patterns')
        ax2.set_title('Vocabulary Expansion Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'language_evolution_fast.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to results directory")

if __name__ == "__main__":
    experiment = FastLanguageEvolution()
    experiment.run_evolution(rounds=50)