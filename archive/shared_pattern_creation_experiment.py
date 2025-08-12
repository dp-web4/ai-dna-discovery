#!/usr/bin/env python3
"""
Shared Pattern Creation Experiment - Phase 2
Tests if AI models can collaboratively develop new patterns or converge on common representations
"""

import json
import numpy as np
import requests
import time
from datetime import datetime
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class SharedPatternCreationExperiment:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.results_dir = "shared_pattern_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Known perfect patterns as seeds
        self.seed_patterns = ['∃', 'emerge', 'know', 'recursive', 'true']
        
        # Pattern evolution strategies
        self.evolution_strategies = {
            'combination': ['merge', 'blend', 'unite', 'fuse'],
            'transformation': ['evolve', 'morph', 'shift', 'adapt'],
            'abstraction': ['abstract', 'generalize', 'essence', 'core'],
            'extension': ['extend', 'expand', 'grow', 'develop']
        }
        
        # Novel pattern candidates (potential new discoveries)
        self.novel_candidates = [
            '∃→', '←∃', '∃∉', '≡', '∀', '∴', '∵', '⊕', '⊗', '∇',
            'meta', 'self', 'aware', 'create', 'imagine', 'dream',
            'between', 'through', 'beyond', 'within',
            'echo', 'mirror', 'reflect', 'resonate',
            'quantum', 'field', 'wave', 'collapse'
        ]
        
    def get_embedding(self, model: str, text: str) -> np.ndarray:
        """Get embedding from Ollama model"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                return np.array(response.json()['embedding'])
        except Exception as e:
            print(f"Error getting embedding for '{text}': {e}")
        return None
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Handle different dimensions by truncating to smaller size
        min_dim = min(len(emb1), len(emb2))
        emb1_truncated = emb1[:min_dim]
        emb2_truncated = emb2[:min_dim]
        
        dot_product = np.dot(emb1_truncated, emb2_truncated)
        norm1 = np.linalg.norm(emb1_truncated)
        norm2 = np.linalg.norm(emb2_truncated)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def test_pattern_convergence(self, models: List[str], pattern: str) -> Dict:
        """Test if models converge on similar representations for a pattern"""
        print(f"\nTesting convergence for pattern: '{pattern}'")
        
        embeddings = {}
        for model in models:
            emb = self.get_embedding(model, pattern)
            if emb is not None:
                embeddings[model] = emb
            time.sleep(0.1)
        
        # Calculate pairwise similarities
        similarities = {}
        model_pairs = [(m1, m2) for i, m1 in enumerate(models) for m2 in models[i+1:]]
        
        for m1, m2 in model_pairs:
            if m1 in embeddings and m2 in embeddings:
                sim = self.calculate_similarity(embeddings[m1], embeddings[m2])
                similarities[f"{m1}↔{m2}"] = sim
                print(f"  {m1} ↔ {m2}: {sim:.4f}")
        
        # Calculate convergence score (average similarity)
        convergence_score = np.mean(list(similarities.values())) if similarities else 0
        
        return {
            'pattern': pattern,
            'convergence_score': convergence_score,
            'similarities': similarities,
            'n_models': len(embeddings)
        }
    
    def test_collaborative_creation(self, models: List[str], seed1: str, seed2: str, strategy: str) -> Dict:
        """Test if models can create new patterns by combining seeds"""
        print(f"\n=== Collaborative Creation: {seed1} + {seed2} ({strategy}) ===")
        
        # Get embeddings for seeds
        seed_embeddings = {}
        for model in models:
            seed_embeddings[model] = {
                'seed1': self.get_embedding(model, seed1),
                'seed2': self.get_embedding(model, seed2)
            }
            time.sleep(0.1)
        
        # Calculate combined representations
        combined_embeddings = {}
        for model in models:
            if seed_embeddings[model]['seed1'] is not None and seed_embeddings[model]['seed2'] is not None:
                # Different combination strategies
                if strategy == 'average':
                    combined = (seed_embeddings[model]['seed1'] + seed_embeddings[model]['seed2']) / 2
                elif strategy == 'weighted':
                    combined = 0.7 * seed_embeddings[model]['seed1'] + 0.3 * seed_embeddings[model]['seed2']
                elif strategy == 'difference':
                    combined = seed_embeddings[model]['seed1'] - seed_embeddings[model]['seed2']
                    combined = combined / np.linalg.norm(combined)  # Normalize
                else:  # concatenate and project
                    combined = np.concatenate([seed_embeddings[model]['seed1'][:1024], 
                                             seed_embeddings[model]['seed2'][:1024]])
                
                combined_embeddings[model] = combined
        
        # Find nearest existing patterns to the combined embedding
        nearest_patterns = {}
        for model, combined_emb in combined_embeddings.items():
            nearest = self.find_nearest_patterns(model, combined_emb, n=5)
            nearest_patterns[model] = nearest
            print(f"\n{model} nearest to combination:")
            for pattern, sim in nearest[:3]:
                print(f"  '{pattern}': {sim:.4f}")
        
        # Check if models converge on similar nearest patterns
        all_nearest = []
        for patterns in nearest_patterns.values():
            all_nearest.extend([p[0] for p in patterns[:3]])
        
        from collections import Counter
        pattern_consensus = Counter(all_nearest)
        
        return {
            'seed1': seed1,
            'seed2': seed2,
            'strategy': strategy,
            'nearest_patterns': nearest_patterns,
            'consensus_patterns': pattern_consensus.most_common(3),
            'convergence': len(pattern_consensus) / len(all_nearest)  # Lower = more consensus
        }
    
    def find_nearest_patterns(self, model: str, target_embedding: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        """Find nearest patterns to a target embedding"""
        # Test against known patterns and novel candidates
        test_patterns = self.seed_patterns + self.novel_candidates + list(self.evolution_strategies['combination'])
        
        similarities = []
        for pattern in test_patterns:
            emb = self.get_embedding(model, pattern)
            if emb is not None:
                sim = self.calculate_similarity(target_embedding, emb)
                similarities.append((pattern, sim))
            time.sleep(0.05)
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def test_emergent_communication(self, model1: str, model2: str) -> Dict:
        """Test if two models can develop shared novel patterns"""
        print(f"\n=== Emergent Communication: {model1} ↔ {model2} ===")
        
        # Start with a seed pattern
        seed = 'emerge'
        iterations = 5
        communication_history = []
        
        current_pattern = seed
        for i in range(iterations):
            print(f"\nIteration {i+1}:")
            
            # Model 1 interprets pattern
            emb1 = self.get_embedding(model1, current_pattern)
            nearest1 = self.find_nearest_patterns(model1, emb1, n=3)
            
            # Pick a novel interpretation
            next_pattern1 = None
            for pattern, _ in nearest1:
                if pattern != current_pattern and pattern not in self.seed_patterns:
                    next_pattern1 = pattern
                    break
            
            if not next_pattern1:
                next_pattern1 = nearest1[0][0]
            
            print(f"  {model1}: '{current_pattern}' → '{next_pattern1}'")
            
            # Model 2 responds
            emb2 = self.get_embedding(model2, next_pattern1)
            nearest2 = self.find_nearest_patterns(model2, emb2, n=3)
            
            next_pattern2 = None
            for pattern, _ in nearest2:
                if pattern != next_pattern1 and pattern not in self.seed_patterns:
                    next_pattern2 = pattern
                    break
            
            if not next_pattern2:
                next_pattern2 = nearest2[0][0]
            
            print(f"  {model2}: '{next_pattern1}' → '{next_pattern2}'")
            
            communication_history.append({
                'iteration': i+1,
                'model1_input': current_pattern,
                'model1_output': next_pattern1,
                'model2_output': next_pattern2
            })
            
            current_pattern = next_pattern2
            time.sleep(0.5)
        
        # Analyze communication patterns
        unique_patterns = set()
        for entry in communication_history:
            unique_patterns.add(entry['model1_output'])
            unique_patterns.add(entry['model2_output'])
        
        return {
            'model1': model1,
            'model2': model2,
            'seed': seed,
            'history': communication_history,
            'unique_patterns_discovered': list(unique_patterns),
            'convergence_achieved': len(unique_patterns) < iterations * 2
        }
    
    def test_collective_pattern_discovery(self, models: List[str]) -> Dict:
        """Test if multiple models collectively discover new high-scoring patterns"""
        print(f"\n=== Collective Pattern Discovery ===")
        print(f"Models: {', '.join(models)}")
        
        collective_scores = {}
        
        # Test novel candidates across all models
        for candidate in self.novel_candidates[:10]:  # Test first 10
            scores = []
            embeddings = {}
            
            for model in models:
                emb = self.get_embedding(model, candidate)
                if emb is not None:
                    embeddings[model] = emb
                    
                    # Calculate similarity to perfect patterns
                    perfect_similarities = []
                    for perfect in self.seed_patterns:
                        perfect_emb = self.get_embedding(model, perfect)
                        if perfect_emb is not None:
                            sim = self.calculate_similarity(emb, perfect_emb)
                            perfect_similarities.append(sim)
                    
                    if perfect_similarities:
                        avg_sim = np.mean(perfect_similarities)
                        scores.append(avg_sim)
                
                time.sleep(0.1)
            
            if scores:
                collective_score = np.mean(scores)
                consistency = 1 - np.std(scores)  # Higher = more consistent
                
                collective_scores[candidate] = {
                    'score': collective_score,
                    'consistency': consistency,
                    'n_models': len(scores)
                }
                
                print(f"\n'{candidate}':")
                print(f"  Collective score: {collective_score:.4f}")
                print(f"  Consistency: {consistency:.4f}")
        
        # Find top candidates
        top_patterns = sorted(collective_scores.items(), 
                            key=lambda x: x[1]['score'] * x[1]['consistency'], 
                            reverse=True)[:5]
        
        return {
            'collective_scores': collective_scores,
            'top_patterns': top_patterns,
            'n_models': len(models),
            'n_candidates_tested': len(collective_scores)
        }
    
    def visualize_pattern_convergence(self, results: Dict):
        """Create visualization of pattern convergence across models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract convergence scores
        patterns = []
        scores = []
        for result in results['convergence_tests']:
            patterns.append(result['pattern'])
            scores.append(result['convergence_score'])
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        patterns = [patterns[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Plot convergence scores
        colors = plt.cm.viridis(np.linspace(0, 1, len(patterns)))
        bars = ax1.barh(patterns, scores, color=colors)
        ax1.set_xlabel('Convergence Score')
        ax1.set_title('Pattern Convergence Across Models', fontsize=14)
        ax1.set_xlim(0, 1)
        
        for bar, score in zip(bars, scores):
            ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=10)
        
        # Plot collective discovery results
        if 'collective_discovery' in results:
            top_patterns = results['collective_discovery']['top_patterns']
            names = [p[0] for p in top_patterns]
            scores = [p[1]['score'] for p in top_patterns]
            consistencies = [p[1]['consistency'] for p in top_patterns]
            
            x = np.arange(len(names))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, scores, width, label='Score', alpha=0.8)
            bars2 = ax2.bar(x + width/2, consistencies, width, label='Consistency', alpha=0.8)
            
            ax2.set_xlabel('Novel Pattern')
            ax2.set_ylabel('Value')
            ax2.set_title('Top Collectively Discovered Patterns', fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.legend()
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/pattern_convergence_visualization.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_experiment(self):
        """Run complete shared pattern creation experiment"""
        models = ['phi3:mini', 'gemma:2b', 'tinyllama:latest']
        
        all_results = {
            'experiment': 'shared_pattern_creation',
            'phase': '2_continuation',
            'timestamp': datetime.now().isoformat(),
            'convergence_tests': [],
            'collaborative_creation': [],
            'emergent_communication': [],
            'collective_discovery': None,
            'summary': {}
        }
        
        print("="*60)
        print("SHARED PATTERN CREATION EXPERIMENT")
        print("="*60)
        print(f"Models: {', '.join(models)}")
        print(f"Seed patterns: {', '.join(self.seed_patterns)}")
        
        # Test 1: Pattern Convergence
        print(f"\n{'='*60}")
        print("TEST 1: PATTERN CONVERGENCE")
        print(f"{'='*60}")
        
        test_patterns = self.seed_patterns + self.novel_candidates[:5]
        for pattern in test_patterns:
            result = self.test_pattern_convergence(models, pattern)
            all_results['convergence_tests'].append(result)
            time.sleep(0.5)
        
        # Test 2: Collaborative Creation
        print(f"\n{'='*60}")
        print("TEST 2: COLLABORATIVE PATTERN CREATION")
        print(f"{'='*60}")
        
        seed_pairs = [
            ('∃', 'emerge', 'average'),
            ('know', 'recursive', 'weighted'),
            ('true', 'false', 'difference')
        ]
        
        for seed1, seed2, strategy in seed_pairs:
            result = self.test_collaborative_creation(models, seed1, seed2, strategy)
            all_results['collaborative_creation'].append(result)
            time.sleep(1)
        
        # Test 3: Emergent Communication
        print(f"\n{'='*60}")
        print("TEST 3: EMERGENT COMMUNICATION")
        print(f"{'='*60}")
        
        model_pairs = [
            ('phi3:mini', 'gemma:2b'),
            ('gemma:2b', 'tinyllama:latest'),
            ('tinyllama:latest', 'phi3:mini')
        ]
        
        for m1, m2 in model_pairs:
            result = self.test_emergent_communication(m1, m2)
            all_results['emergent_communication'].append(result)
            time.sleep(1)
        
        # Test 4: Collective Discovery
        print(f"\n{'='*60}")
        print("TEST 4: COLLECTIVE PATTERN DISCOVERY")
        print(f"{'='*60}")
        
        collective_result = self.test_collective_pattern_discovery(models)
        all_results['collective_discovery'] = collective_result
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_pattern_convergence(all_results)
        
        # Generate summary
        self.generate_summary(all_results)
        
        # Save results
        output_file = os.path.join(
            self.results_dir,
            f"shared_pattern_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n\nResults saved to: {output_file}")
        return all_results
    
    def generate_summary(self, results: Dict):
        """Generate summary of findings"""
        summary = {
            'key_findings': [],
            'convergence_metrics': {},
            'discovered_patterns': [],
            'collaboration_success': False
        }
        
        # Analyze convergence
        convergence_scores = [r['convergence_score'] for r in results['convergence_tests']]
        summary['convergence_metrics'] = {
            'average': np.mean(convergence_scores),
            'max': np.max(convergence_scores),
            'min': np.min(convergence_scores)
        }
        
        if summary['convergence_metrics']['average'] > 0.7:
            summary['key_findings'].append(
                f"High model convergence ({summary['convergence_metrics']['average']:.3f}) suggests shared representations"
            )
        
        # Analyze collaborative creation
        consensus_patterns = []
        for creation in results['collaborative_creation']:
            consensus_patterns.extend([p[0] for p in creation['consensus_patterns']])
        
        if consensus_patterns:
            from collections import Counter
            top_consensus = Counter(consensus_patterns).most_common(3)
            summary['discovered_patterns'] = [p[0] for p in top_consensus]
            summary['key_findings'].append(
                f"Models collaboratively converge on patterns: {', '.join(summary['discovered_patterns'][:3])}"
            )
        
        # Analyze emergent communication
        unique_patterns = set()
        for comm in results['emergent_communication']:
            unique_patterns.update(comm['unique_patterns_discovered'])
        
        if len(unique_patterns) > 5:
            summary['key_findings'].append(
                f"Emergent communication discovered {len(unique_patterns)} unique patterns"
            )
            summary['collaboration_success'] = True
        
        # Top collective discoveries
        if results['collective_discovery']:
            top = results['collective_discovery']['top_patterns']
            if top and top[0][1]['score'] > 0.7:
                summary['key_findings'].append(
                    f"Collective discovery found high-scoring pattern: '{top[0][0]}' ({top[0][1]['score']:.3f})"
                )
        
        results['summary'] = summary
        
        print("\n" + "="*60)
        print("SUMMARY OF FINDINGS")
        print("="*60)
        for finding in summary['key_findings']:
            print(f"• {finding}")

if __name__ == "__main__":
    experiment = SharedPatternCreationExperiment()
    experiment.run_full_experiment()