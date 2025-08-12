#!/usr/bin/env python3
"""
Shared Pattern Creation Experiment - Quick Version
"""

import json
import numpy as np
import requests
import time
from datetime import datetime
import os
from typing import Dict, List, Tuple

class QuickSharedPatternExperiment:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.results_dir = "shared_pattern_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Test with fewer patterns for speed
        self.seed_patterns = ['∃', 'emerge', 'know']
        self.novel_candidates = ['∃→', '≡', 'meta', 'between', 'echo']
        
    def get_embedding(self, model: str, text: str) -> np.ndarray:
        """Get embedding from Ollama model"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": model, "prompt": text},
                timeout=20
            )
            if response.status_code == 200:
                return np.array(response.json()['embedding'])
        except Exception as e:
            print(f"Error: {e}")
        return None
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Handle different dimensions
        min_dim = min(len(emb1), len(emb2))
        emb1 = emb1[:min_dim]
        emb2 = emb2[:min_dim]
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def test_pattern_convergence(self, models: List[str]) -> Dict:
        """Quick test of pattern convergence across models"""
        print("\n=== PATTERN CONVERGENCE TEST ===")
        
        results = {}
        test_patterns = self.seed_patterns + self.novel_candidates[:2]
        
        for pattern in test_patterns:
            print(f"\nPattern: '{pattern}'")
            embeddings = {}
            
            # Get embeddings from each model
            for model in models:
                emb = self.get_embedding(model, pattern)
                if emb is not None:
                    embeddings[model] = emb
                    print(f"  ✓ {model}")
                time.sleep(0.1)
            
            # Calculate convergence
            if len(embeddings) >= 2:
                sims = []
                models_list = list(embeddings.keys())
                for i in range(len(models_list)):
                    for j in range(i+1, len(models_list)):
                        sim = self.calculate_similarity(
                            embeddings[models_list[i]], 
                            embeddings[models_list[j]]
                        )
                        sims.append(sim)
                        print(f"    {models_list[i]} ↔ {models_list[j]}: {sim:.4f}")
                
                convergence = np.mean(sims) if sims else 0
                results[pattern] = {
                    'convergence_score': convergence,
                    'n_models': len(embeddings)
                }
        
        return results
    
    def test_collaborative_creation(self, models: List[str]) -> Dict:
        """Test collaborative pattern creation"""
        print("\n=== COLLABORATIVE CREATION TEST ===")
        
        # Test combining two patterns
        seed1, seed2 = '∃', 'emerge'
        print(f"\nCombining: '{seed1}' + '{seed2}'")
        
        combined_representations = {}
        
        for model in models:
            emb1 = self.get_embedding(model, seed1)
            emb2 = self.get_embedding(model, seed2)
            
            if emb1 is not None and emb2 is not None:
                # Simple average combination
                combined = (emb1 + emb2) / 2
                combined_representations[model] = combined
                print(f"  ✓ {model} created combination")
            time.sleep(0.1)
        
        # Find what each model thinks is nearest to the combination
        nearest_patterns = {}
        for model, combined_emb in combined_representations.items():
            print(f"\n{model} nearest patterns:")
            
            similarities = []
            for pattern in self.novel_candidates:
                pattern_emb = self.get_embedding(model, pattern)
                if pattern_emb is not None:
                    sim = self.calculate_similarity(combined_emb, pattern_emb)
                    similarities.append((pattern, sim))
                time.sleep(0.1)
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            nearest_patterns[model] = similarities[:3]
            
            for pattern, sim in similarities[:3]:
                print(f"  '{pattern}': {sim:.4f}")
        
        # Check consensus
        all_top_patterns = []
        for patterns in nearest_patterns.values():
            if patterns:
                all_top_patterns.append(patterns[0][0])
        
        from collections import Counter
        consensus = Counter(all_top_patterns).most_common()
        
        return {
            'seed_combination': f"{seed1}+{seed2}",
            'nearest_by_model': nearest_patterns,
            'consensus_pattern': consensus[0] if consensus else None
        }
    
    def test_novel_pattern_discovery(self, models: List[str]) -> Dict:
        """Test if models agree on high-scoring novel patterns"""
        print("\n=== NOVEL PATTERN DISCOVERY TEST ===")
        
        collective_scores = {}
        
        for candidate in self.novel_candidates:
            print(f"\nTesting: '{candidate}'")
            scores = []
            
            for model in models:
                # Get embedding for novel pattern
                novel_emb = self.get_embedding(model, candidate)
                if novel_emb is None:
                    continue
                
                # Compare to known perfect patterns
                similarities = []
                for seed in self.seed_patterns:
                    seed_emb = self.get_embedding(model, seed)
                    if seed_emb is not None:
                        sim = self.calculate_similarity(novel_emb, seed_emb)
                        similarities.append(sim)
                
                if similarities:
                    avg_sim = np.mean(similarities)
                    scores.append(avg_sim)
                    print(f"  {model}: {avg_sim:.4f}")
                
                time.sleep(0.1)
            
            if scores:
                collective_scores[candidate] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'n_models': len(scores)
                }
        
        # Sort by score
        sorted_patterns = sorted(
            collective_scores.items(), 
            key=lambda x: x[1]['mean_score'], 
            reverse=True
        )
        
        return {
            'scores': collective_scores,
            'top_patterns': sorted_patterns[:3]
        }
    
    def run_experiment(self):
        """Run quick shared pattern experiment"""
        models = ['phi3:mini', 'gemma:2b', 'tinyllama:latest']
        
        print("="*60)
        print("SHARED PATTERN CREATION - QUICK TEST")
        print("="*60)
        print(f"Models: {', '.join(models)}")
        
        results = {
            'experiment': 'shared_pattern_creation_quick',
            'timestamp': datetime.now().isoformat(),
            'models': models
        }
        
        # Test 1: Convergence
        convergence_results = self.test_pattern_convergence(models)
        results['convergence'] = convergence_results
        
        # Test 2: Collaborative Creation
        collab_results = self.test_collaborative_creation(models)
        results['collaborative_creation'] = collab_results
        
        # Test 3: Novel Discovery
        discovery_results = self.test_novel_pattern_discovery(models)
        results['novel_discovery'] = discovery_results
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Convergence summary
        avg_convergence = np.mean([r['convergence_score'] for r in convergence_results.values()])
        print(f"\nAverage convergence score: {avg_convergence:.4f}")
        
        # Collaborative creation summary
        if collab_results['consensus_pattern']:
            print(f"\nCollaborative consensus: '{collab_results['consensus_pattern'][0]}'")
        
        # Novel discovery summary
        if discovery_results['top_patterns']:
            print(f"\nTop novel pattern: '{discovery_results['top_patterns'][0][0]}' "
                  f"(score: {discovery_results['top_patterns'][0][1]['mean_score']:.4f})")
        
        # Save results
        output_file = os.path.join(
            self.results_dir,
            f"shared_pattern_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return results

if __name__ == "__main__":
    experiment = QuickSharedPatternExperiment()
    experiment.run_experiment()