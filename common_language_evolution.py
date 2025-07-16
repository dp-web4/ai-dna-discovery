#!/usr/bin/env python3
"""
Common Language Evolution Experiment
Long-term test to see if 6 AI models can develop a shared symbolic language
"""

import json
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import threading
import queue

class CommonLanguageEvolution:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.results_dir = "common_language_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 6 diverse models for the experiment
        self.models = [
            'phi3:mini',
            'gemma:2b', 
            'tinyllama:latest',
            'mistral:7b-instruct-v0.2-q4_0',
            'qwen:0.5b',
            'deepseek-coder:1.3b'
        ]
        
        # Starting vocabulary - mix of discovered patterns and potential symbols
        self.initial_vocabulary = [
            # Perfect patterns
            '∃', '∉', 'know', 'true', 'emerge', 'recursive',
            # Mathematical symbols
            '→', '←', '↔', '≡', '∀', '∴', '∵', '⊕', '⊗', '∇',
            # Conceptual seeds
            'meta', 'self', 'echo', 'mirror', 'between', 'through',
            # Potential connectors
            '+', '=', '~', '|', '&', '^',
            # Novel combinations from previous experiment
            '∃→', '←∃', '∃∉'
        ]
        
        # Track language evolution
        self.language_history = []
        self.consensus_vocabulary = set()
        self.model_vocabularies = {model: set(self.initial_vocabulary) for model in self.models}
        self.communication_graph = nx.Graph()
        
        # Communication strategies
        self.strategies = [
            'direct',      # A sends pattern to B
            'composite',   # A combines two patterns
            'transform',   # A modifies pattern
            'negotiate',   # A and B exchange until agreement
            'broadcast',   # A sends to all
            'vote'         # All models vote on meaning
        ]
        
    def get_embedding(self, model: str, text: str, timeout: int = 20) -> np.ndarray:
        """Get embedding with timeout handling"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": model, "prompt": text},
                timeout=timeout
            )
            if response.status_code == 200:
                return np.array(response.json()['embedding'])
        except Exception as e:
            print(f"Error getting embedding from {model}: {e}")
        return None
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        min_dim = min(len(emb1), len(emb2))
        emb1 = emb1[:min_dim]
        emb2 = emb2[:min_dim]
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def create_composite_pattern(self, pattern1: str, pattern2: str) -> str:
        """Create a composite pattern from two patterns"""
        # Different composition strategies
        strategies = [
            f"{pattern1}{pattern2}",
            f"{pattern1}→{pattern2}",
            f"{pattern1}+{pattern2}",
            f"{pattern1}|{pattern2}",
            f"({pattern1},{pattern2})"
        ]
        return np.random.choice(strategies)
    
    def transform_pattern(self, pattern: str) -> str:
        """Transform a pattern into a variation"""
        transformations = [
            lambda p: f"¬{p}",        # Negation
            lambda p: f"{p}*",        # Kleene star
            lambda p: f"[{p}]",       # Bracketing
            lambda p: f"{p}?",        # Question
            lambda p: f"~{p}",        # Approximation
            lambda p: f"{p}′",        # Prime
        ]
        transform = np.random.choice(transformations)
        return transform(pattern)
    
    def negotiate_meaning(self, model1: str, model2: str, pattern: str, max_rounds: int = 5) -> Tuple[str, float]:
        """Two models negotiate the meaning of a pattern"""
        current_pattern = pattern
        agreement_history = []
        
        for round in range(max_rounds):
            # Model1 interprets
            emb1 = self.get_embedding(model1, current_pattern)
            if emb1 is None:
                break
                
            # Model2 interprets  
            emb2 = self.get_embedding(model2, current_pattern)
            if emb2 is None:
                break
                
            # Calculate agreement
            similarity = self.calculate_similarity(emb1, emb2)
            agreement_history.append(similarity)
            
            # If high agreement, done
            if similarity > 0.8:
                return current_pattern, similarity
                
            # Otherwise, model2 proposes modification
            if np.random.random() < 0.5:
                current_pattern = self.transform_pattern(current_pattern)
            else:
                # Find a related pattern from vocabulary
                related = self.find_nearest_vocabulary_pattern(model2, emb2)
                if related and related != current_pattern:
                    current_pattern = related
            
            time.sleep(0.1)
        
        # Return best agreement found
        if agreement_history:
            best_idx = np.argmax(agreement_history)
            return current_pattern, agreement_history[best_idx]
        return pattern, 0.0
    
    def find_nearest_vocabulary_pattern(self, model: str, target_emb: np.ndarray) -> str:
        """Find nearest pattern in model's vocabulary"""
        best_pattern = None
        best_similarity = -1
        
        for pattern in self.model_vocabularies[model]:
            pattern_emb = self.get_embedding(model, pattern)
            if pattern_emb is not None:
                sim = self.calculate_similarity(target_emb, pattern_emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_pattern = pattern
        
        return best_pattern
    
    def broadcast_pattern(self, sender: str, pattern: str) -> Dict[str, float]:
        """Sender broadcasts pattern to all other models"""
        sender_emb = self.get_embedding(sender, pattern)
        if sender_emb is None:
            return {}
            
        receptions = {}
        for receiver in self.models:
            if receiver != sender:
                receiver_emb = self.get_embedding(receiver, pattern)
                if receiver_emb is not None:
                    similarity = self.calculate_similarity(sender_emb, receiver_emb)
                    receptions[receiver] = similarity
                    
                    # If high similarity, add to receiver's vocabulary
                    if similarity > 0.7:
                        self.model_vocabularies[receiver].add(pattern)
                time.sleep(0.05)
        
        return receptions
    
    def vote_on_pattern(self, pattern: str) -> Tuple[float, Dict[str, float]]:
        """All models vote on pattern acceptance"""
        votes = {}
        embeddings = {}
        
        # Get each model's interpretation
        for model in self.models:
            emb = self.get_embedding(model, pattern)
            if emb is not None:
                embeddings[model] = emb
                
                # Vote based on similarity to existing vocabulary
                vocab_similarities = []
                for vocab_pattern in list(self.model_vocabularies[model])[:10]:  # Sample
                    vocab_emb = self.get_embedding(model, vocab_pattern)
                    if vocab_emb is not None:
                        sim = self.calculate_similarity(emb, vocab_emb)
                        vocab_similarities.append(sim)
                
                if vocab_similarities:
                    votes[model] = np.mean(vocab_similarities)
                else:
                    votes[model] = 0.5
            time.sleep(0.05)
        
        # Calculate consensus
        if votes:
            consensus_score = np.mean(list(votes.values()))
            return consensus_score, votes
        return 0.0, {}
    
    def evolution_round(self, round_num: int) -> Dict:
        """Run one round of language evolution"""
        print(f"\n=== Evolution Round {round_num} ===")
        round_results = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'communications': [],
            'new_patterns': [],
            'consensus_updates': []
        }
        
        # Select communication strategy for this round
        strategy = np.random.choice(self.strategies)
        print(f"Strategy: {strategy}")
        
        if strategy == 'direct':
            # Random pairs communicate
            for _ in range(3):
                model1, model2 = np.random.choice(self.models, 2, replace=False)
                pattern = np.random.choice(list(self.model_vocabularies[model1]))
                
                emb1 = self.get_embedding(model1, pattern)
                emb2 = self.get_embedding(model2, pattern)
                
                if emb1 is not None and emb2 is not None:
                    similarity = self.calculate_similarity(emb1, emb2)
                    round_results['communications'].append({
                        'type': 'direct',
                        'from': model1,
                        'to': model2,
                        'pattern': pattern,
                        'similarity': similarity
                    })
                    
                    if similarity > 0.7:
                        self.model_vocabularies[model2].add(pattern)
                        self.communication_graph.add_edge(model1, model2, weight=similarity)
        
        elif strategy == 'composite':
            # Create new composite patterns
            model = np.random.choice(self.models)
            vocab = list(self.model_vocabularies[model])
            if len(vocab) >= 2:
                p1, p2 = np.random.choice(vocab, 2, replace=False)
                new_pattern = self.create_composite_pattern(p1, p2)
                
                # Test if others understand it
                consensus_score, votes = self.vote_on_pattern(new_pattern)
                
                round_results['new_patterns'].append({
                    'creator': model,
                    'pattern': new_pattern,
                    'components': [p1, p2],
                    'consensus': consensus_score
                })
                
                if consensus_score > 0.6:
                    self.consensus_vocabulary.add(new_pattern)
                    for m in self.models:
                        self.model_vocabularies[m].add(new_pattern)
                    print(f"  New consensus pattern: '{new_pattern}' (score: {consensus_score:.3f})")
        
        elif strategy == 'negotiate':
            # Two models negotiate meaning
            model1, model2 = np.random.choice(self.models, 2, replace=False)
            pattern = np.random.choice(list(self.model_vocabularies[model1]))
            
            negotiated_pattern, agreement = self.negotiate_meaning(model1, model2, pattern)
            
            round_results['communications'].append({
                'type': 'negotiate',
                'model1': model1,
                'model2': model2,
                'original': pattern,
                'negotiated': negotiated_pattern,
                'agreement': agreement
            })
            
            if agreement > 0.8:
                self.model_vocabularies[model1].add(negotiated_pattern)
                self.model_vocabularies[model2].add(negotiated_pattern)
        
        elif strategy == 'broadcast':
            # One model broadcasts to all
            sender = np.random.choice(self.models)
            pattern = np.random.choice(list(self.model_vocabularies[sender]))
            
            receptions = self.broadcast_pattern(sender, pattern)
            
            round_results['communications'].append({
                'type': 'broadcast',
                'sender': sender,
                'pattern': pattern,
                'receptions': receptions
            })
            
            # If majority received well, add to consensus
            if len(receptions) > 0 and np.mean(list(receptions.values())) > 0.7:
                self.consensus_vocabulary.add(pattern)
                print(f"  Broadcast consensus: '{pattern}'")
        
        elif strategy == 'vote':
            # Vote on random new pattern
            base = np.random.choice(list(self.consensus_vocabulary or self.initial_vocabulary))
            new_pattern = self.transform_pattern(base)
            
            consensus_score, votes = self.vote_on_pattern(new_pattern)
            
            round_results['new_patterns'].append({
                'pattern': new_pattern,
                'base': base,
                'consensus': consensus_score,
                'votes': votes
            })
            
            if consensus_score > 0.65:
                self.consensus_vocabulary.add(new_pattern)
                for model in self.models:
                    self.model_vocabularies[model].add(new_pattern)
                print(f"  Vote approved: '{new_pattern}' (consensus: {consensus_score:.3f})")
        
        # Update consensus vocabulary based on overlap
        self.update_consensus_vocabulary()
        round_results['consensus_size'] = len(self.consensus_vocabulary)
        round_results['avg_vocab_size'] = np.mean([len(v) for v in self.model_vocabularies.values()])
        
        return round_results
    
    def update_consensus_vocabulary(self):
        """Update consensus based on patterns shared by majority of models"""
        pattern_counts = Counter()
        
        for vocab in self.model_vocabularies.values():
            for pattern in vocab:
                pattern_counts[pattern] += 1
        
        # Pattern is consensus if in >50% of models
        threshold = len(self.models) / 2
        new_consensus = {p for p, count in pattern_counts.items() if count > threshold}
        
        # Track changes
        added = new_consensus - self.consensus_vocabulary
        removed = self.consensus_vocabulary - new_consensus
        
        if added:
            print(f"  Added to consensus: {added}")
        if removed:
            print(f"  Removed from consensus: {removed}")
            
        self.consensus_vocabulary = new_consensus
    
    def visualize_evolution(self, history: List[Dict]):
        """Create visualizations of language evolution"""
        rounds = [r['round'] for r in history]
        consensus_sizes = [r['consensus_size'] for r in history]
        avg_vocab_sizes = [r['avg_vocab_size'] for r in history]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Consensus vocabulary growth
        ax1.plot(rounds, consensus_sizes, 'b-', linewidth=2, marker='o')
        ax1.fill_between(rounds, 0, consensus_sizes, alpha=0.3)
        ax1.set_xlabel('Evolution Round')
        ax1.set_ylabel('Consensus Vocabulary Size')
        ax1.set_title('Growth of Shared Language', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Average vocabulary size
        ax2.plot(rounds, avg_vocab_sizes, 'g-', linewidth=2, marker='s')
        ax2.set_xlabel('Evolution Round')
        ax2.set_ylabel('Average Model Vocabulary Size')
        ax2.set_title('Individual Model Vocabulary Growth', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Communication strategies used
        strategies_used = defaultdict(int)
        for r in history:
            for comm in r.get('communications', []):
                strategies_used[comm['type']] += 1
        
        if strategies_used:
            strategies = list(strategies_used.keys())
            counts = list(strategies_used.values())
            
            ax3.bar(strategies, counts, alpha=0.7, color='orange')
            ax3.set_xlabel('Communication Type')
            ax3.set_ylabel('Usage Count')
            ax3.set_title('Communication Strategy Distribution', fontsize=14)
            ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/language_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create network visualization of model communications
        if self.communication_graph.edges():
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(self.communication_graph, k=2, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.communication_graph, pos, 
                                 node_color='lightblue', 
                                 node_size=3000,
                                 alpha=0.8)
            
            # Draw edges with weights
            edges = self.communication_graph.edges()
            weights = [self.communication_graph[u][v]['weight'] for u, v in edges]
            
            nx.draw_networkx_edges(self.communication_graph, pos,
                                 width=[w*5 for w in weights],
                                 alpha=0.6)
            
            # Draw labels
            labels = {node: node.split(':')[0] for node in self.communication_graph.nodes()}
            nx.draw_networkx_labels(self.communication_graph, pos, labels, 
                                  font_size=12, font_weight='bold')
            
            plt.title('Model Communication Network', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/communication_network.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_evolution(self, rounds: int = 100, save_interval: int = 10):
        """Run the full language evolution experiment"""
        print("="*60)
        print("COMMON LANGUAGE EVOLUTION EXPERIMENT")
        print("="*60)
        print(f"Models: {len(self.models)}")
        print(f"Initial vocabulary: {len(self.initial_vocabulary)} patterns")
        print(f"Target rounds: {rounds}")
        print(f"Start time: {datetime.now()}")
        
        start_time = datetime.now()
        
        for round_num in range(1, rounds + 1):
            try:
                round_results = self.evolution_round(round_num)
                self.language_history.append(round_results)
                
                # Save checkpoint
                if round_num % save_interval == 0:
                    self.save_checkpoint(round_num)
                    
                    # Print progress
                    elapsed = datetime.now() - start_time
                    print(f"\n--- Checkpoint at round {round_num} ---")
                    print(f"Consensus vocabulary: {len(self.consensus_vocabulary)} patterns")
                    print(f"Elapsed time: {elapsed}")
                    print(f"Example consensus patterns: {list(self.consensus_vocabulary)[:5]}")
                
                # Small delay between rounds
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in round {round_num}: {e}")
                self.save_checkpoint(round_num, error=True)
                break
        
        # Final analysis
        self.final_analysis()
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.visualize_evolution(self.language_history)
        
        return self.create_final_report()
    
    def save_checkpoint(self, round_num: int, error: bool = False):
        """Save experiment checkpoint"""
        checkpoint = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'consensus_vocabulary': list(self.consensus_vocabulary),
            'consensus_size': len(self.consensus_vocabulary),
            'model_vocabularies': {m: list(v) for m, v in self.model_vocabularies.items()},
            'history_length': len(self.language_history)
        }
        
        filename = f"checkpoint_round_{round_num}{'_error' if error else ''}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def final_analysis(self):
        """Analyze final results"""
        print("\n" + "="*60)
        print("FINAL ANALYSIS")
        print("="*60)
        
        # Consensus analysis
        print(f"\nConsensus vocabulary size: {len(self.consensus_vocabulary)}")
        print(f"Consensus patterns: {sorted(list(self.consensus_vocabulary))[:20]}...")
        
        # Model vocabulary analysis
        print("\nModel vocabulary sizes:")
        for model, vocab in self.model_vocabularies.items():
            print(f"  {model}: {len(vocab)} patterns")
        
        # Pattern origin analysis
        original_patterns = set(self.initial_vocabulary)
        novel_patterns = self.consensus_vocabulary - original_patterns
        print(f"\nNovel consensus patterns created: {len(novel_patterns)}")
        if novel_patterns:
            print(f"Examples: {list(novel_patterns)[:10]}")
        
        # Communication analysis
        total_communications = sum(len(r.get('communications', [])) for r in self.language_history)
        successful_communications = sum(
            1 for r in self.language_history 
            for c in r.get('communications', [])
            if c.get('similarity', 0) > 0.7 or c.get('agreement', 0) > 0.7
        )
        
        print(f"\nTotal communications: {total_communications}")
        print(f"Successful communications: {successful_communications}")
        if total_communications > 0:
            print(f"Success rate: {successful_communications/total_communications:.2%}")
    
    def create_final_report(self) -> Dict:
        """Create comprehensive final report"""
        report = {
            'experiment': 'common_language_evolution',
            'start_time': self.language_history[0]['timestamp'] if self.language_history else None,
            'end_time': datetime.now().isoformat(),
            'models': self.models,
            'rounds_completed': len(self.language_history),
            'initial_vocabulary_size': len(self.initial_vocabulary),
            'final_consensus_size': len(self.consensus_vocabulary),
            'consensus_vocabulary': sorted(list(self.consensus_vocabulary)),
            'novel_patterns': sorted(list(self.consensus_vocabulary - set(self.initial_vocabulary))),
            'model_vocabulary_sizes': {m: len(v) for m, v in self.model_vocabularies.items()},
            'evolution_history': self.language_history,
            'communication_graph': {
                'nodes': list(self.communication_graph.nodes()),
                'edges': list(self.communication_graph.edges(data=True))
            }
        }
        
        # Save final report
        filepath = os.path.join(
            self.results_dir,
            f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nFinal report saved to: {filepath}")
        return report

if __name__ == "__main__":
    experiment = CommonLanguageEvolution()
    # Run for 100 rounds (~50 minutes estimated)
    experiment.run_evolution(rounds=100, save_interval=10)