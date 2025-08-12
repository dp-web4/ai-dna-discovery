#!/usr/bin/env python3
"""
AI Handshake Protocol - Based on Grok's recommendation
Test if models can establish shared language through iterative handshaking
"""

import json
import numpy as np
import requests
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

class HandshakeProtocol:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.chat_url = "http://localhost:11434/api/generate"
        self.results_dir = "handshake_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Use all 6 models for comprehensive testing
        self.models = [
            'phi3:mini', 'gemma:2b', 'tinyllama:latest',
            'qwen2:0.5b', 'deepseek-coder:1.3b', 'llama3.2:1b'
        ]
        
        # Track handshake state for each model pair
        self.handshake_states = {}
        self.convergence_history = []
        self.shared_vocabulary = set()
        
        # Handshake parameters
        self.max_iterations = 100
        self.convergence_threshold = 0.8
        self.stability_rounds = 5
        
    def get_embedding(self, model: str, text: str) -> np.ndarray:
        """Get embedding from model"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": model, "prompt": text},
                timeout=15
            )
            if response.status_code == 200:
                return np.array(response.json()['embedding'])
        except Exception as e:
            print(f"  Error getting embedding from {model}: {e}")
        return None
    
    def generate_response(self, model: str, prompt: str) -> str:
        """Get text response from model"""
        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": model, 
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 50}
                },
                timeout=20
            )
            if response.status_code == 200:
                return response.json()['response'].strip()
        except Exception as e:
            print(f"  Error getting response from {model}: {e}")
        return ""
    
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
    
    def initialize_handshake(self, model_a: str, model_b: str):
        """Initialize handshake between two models"""
        print(f"\\n=== Initializing handshake: {model_a} ↔ {model_b} ===")
        
        # Start with AI DNA seed pattern
        seed_pattern = "∃"
        
        # Model A proposes interpretation
        prompt_a = f"Given the symbol '{seed_pattern}', respond with a single word or symbol that captures its essence:"
        response_a = self.generate_response(model_a, prompt_a)
        
        # Model B responds to A's interpretation
        prompt_b = f"Given '{seed_pattern}' and the interpretation '{response_a}', respond with a single word or symbol:"
        response_b = self.generate_response(model_b, prompt_b)
        
        return {
            'seed': seed_pattern,
            'model_a_response': response_a,
            'model_b_response': response_b,
            'iteration': 0,
            'convergence_score': 0.0
        }
    
    def handshake_iteration(self, model_a: str, model_b: str, state: dict):
        """Perform one handshake iteration"""
        iteration = state['iteration'] + 1
        
        # Create combined pattern from previous responses
        if state['model_a_response'] and state['model_b_response']:
            combined_pattern = f"{state['model_a_response']}→{state['model_b_response']}"
        else:
            combined_pattern = state['seed']
        
        print(f"  Iteration {iteration}: Testing '{combined_pattern}'")
        
        # Both models respond to combined pattern
        prompt = f"Given the pattern '{combined_pattern}', respond with a refined single word or symbol:"
        
        response_a = self.generate_response(model_a, prompt)
        response_b = self.generate_response(model_b, prompt)
        
        print(f"    {model_a}: '{response_a}'")
        print(f"    {model_b}: '{response_b}'")
        
        # Calculate convergence using both text and embedding similarity
        text_similarity = 1.0 if response_a.lower() == response_b.lower() else 0.0
        
        emb_a = self.get_embedding(model_a, combined_pattern)
        emb_b = self.get_embedding(model_b, combined_pattern)
        embedding_similarity = self.calculate_similarity(emb_a, emb_b)
        
        # Combined convergence score (weighted)
        convergence_score = 0.7 * embedding_similarity + 0.3 * text_similarity
        
        print(f"    Convergence: {convergence_score:.4f} (text: {text_similarity:.1f}, emb: {embedding_similarity:.4f})")
        
        # Update state
        state.update({
            'iteration': iteration,
            'model_a_response': response_a,
            'model_b_response': response_b,
            'combined_pattern': combined_pattern,
            'convergence_score': convergence_score,
            'text_similarity': text_similarity,
            'embedding_similarity': embedding_similarity,
            'timestamp': datetime.now().isoformat()
        })
        
        return state
    
    def run_pairwise_handshake(self, model_a: str, model_b: str):
        """Run complete handshake between two models"""
        print(f"\\n{'='*60}")
        print(f"HANDSHAKE: {model_a} ↔ {model_b}")
        print(f"{'='*60}")
        
        # Initialize
        state = self.initialize_handshake(model_a, model_b)
        history = [state.copy()]
        
        # Iterate until convergence or max iterations
        converged = False
        stable_rounds = 0
        
        for i in range(self.max_iterations):
            try:
                state = self.handshake_iteration(model_a, model_b, state)
                history.append(state.copy())
                
                # Check for convergence
                if state['convergence_score'] >= self.convergence_threshold:
                    stable_rounds += 1
                    if stable_rounds >= self.stability_rounds:
                        print(f"  ✓ CONVERGENCE ACHIEVED after {state['iteration']} iterations")
                        converged = True
                        break
                else:
                    stable_rounds = 0
                
                # Check for exact text match (early termination)
                if state['text_similarity'] == 1.0:
                    print(f"  ★ EXACT TEXT MATCH achieved!")
                    # Continue for a few more rounds to confirm stability
                    stable_rounds += 1
                
                time.sleep(0.5)  # Brief pause between iterations
                
            except Exception as e:
                print(f"  Error in iteration {i+1}: {e}")
                break
        
        # Final summary
        final_score = state['convergence_score']
        print(f"\\n--- Handshake Summary ---")
        print(f"Iterations: {state['iteration']}")
        print(f"Final convergence: {final_score:.4f}")
        print(f"Converged: {'YES' if converged else 'NO'}")
        print(f"Final pattern: '{state.get('combined_pattern', 'N/A')}'")
        print(f"Final responses: '{state['model_a_response']}' / '{state['model_b_response']}'")
        
        # Save results
        pair_id = f"{model_a.split(':')[0]}_{model_b.split(':')[0]}"
        results_file = os.path.join(self.results_dir, f"handshake_{pair_id}.json")
        
        results = {
            'model_a': model_a,
            'model_b': model_b,
            'converged': converged,
            'final_convergence_score': final_score,
            'iterations': state['iteration'],
            'history': history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_full_handshake_matrix(self):
        """Run handshakes between all model pairs"""
        print(f"\\n{'='*80}")
        print(f"FULL HANDSHAKE PROTOCOL - {len(self.models)} MODELS")
        print(f"{'='*80}")
        print(f"Models: {', '.join([m.split(':')[0] for m in self.models])}")
        print(f"Expected pairs: {len(self.models) * (len(self.models) - 1) // 2}")
        print(f"Max iterations per pair: {self.max_iterations}")
        print(f"Convergence threshold: {self.convergence_threshold}")
        
        start_time = datetime.now()
        all_results = {}
        convergence_matrix = np.zeros((len(self.models), len(self.models)))
        
        # Test all unique pairs
        for i, model_a in enumerate(self.models):
            for j, model_b in enumerate(self.models):
                if i < j:  # Only test each pair once
                    try:
                        results = self.run_pairwise_handshake(model_a, model_b)
                        pair_key = f"{model_a}_{model_b}"
                        all_results[pair_key] = results
                        
                        # Store convergence score in matrix
                        score = results['final_convergence_score']
                        convergence_matrix[i, j] = score
                        convergence_matrix[j, i] = score  # Symmetric
                        
                    except Exception as e:
                        print(f"Error in handshake {model_a} ↔ {model_b}: {e}")
                        continue
        
        # Self-similarity (should be 1.0)
        np.fill_diagonal(convergence_matrix, 1.0)
        
        # Save comprehensive results
        final_results = {
            'experiment': 'handshake_protocol',
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'models': self.models,
            'parameters': {
                'max_iterations': self.max_iterations,
                'convergence_threshold': self.convergence_threshold,
                'stability_rounds': self.stability_rounds
            },
            'pairwise_results': all_results,
            'convergence_matrix': convergence_matrix.tolist(),
            'summary': self.generate_summary(all_results, convergence_matrix)
        }
        
        # Save final results
        final_file = os.path.join(self.results_dir, f"handshake_full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(final_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Create visualizations
        self.create_visualizations(convergence_matrix, all_results)
        
        # Print final summary
        self.print_final_summary(final_results)
        
        return final_results
    
    def generate_summary(self, all_results: dict, convergence_matrix: np.ndarray):
        """Generate experiment summary"""
        n_pairs = len(all_results)
        converged_pairs = sum(1 for r in all_results.values() if r['converged'])
        avg_convergence = np.mean([r['final_convergence_score'] for r in all_results.values()])
        avg_iterations = np.mean([r['iterations'] for r in all_results.values()])
        
        # Find best and worst pairs
        best_pair = max(all_results.items(), key=lambda x: x[1]['final_convergence_score'])
        worst_pair = min(all_results.items(), key=lambda x: x[1]['final_convergence_score'])
        
        return {
            'total_pairs': n_pairs,
            'converged_pairs': converged_pairs,
            'convergence_rate': converged_pairs / n_pairs if n_pairs > 0 else 0,
            'average_convergence_score': avg_convergence,
            'average_iterations': avg_iterations,
            'best_pair': {
                'models': best_pair[0],
                'score': best_pair[1]['final_convergence_score'],
                'iterations': best_pair[1]['iterations']
            },
            'worst_pair': {
                'models': worst_pair[0],
                'score': worst_pair[1]['final_convergence_score'],
                'iterations': worst_pair[1]['iterations']
            }
        }
    
    def create_visualizations(self, convergence_matrix: np.ndarray, all_results: dict):
        """Create visualization plots"""
        model_names = [m.split(':')[0] for m in self.models]
        
        # 1. Convergence matrix heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(convergence_matrix, 
                   xticklabels=model_names, 
                   yticklabels=model_names,
                   annot=True, 
                   cmap='RdYlGn', 
                   center=0.5,
                   vmin=0, 
                   vmax=1,
                   square=True)
        plt.title('Handshake Convergence Matrix', fontsize=16)
        plt.xlabel('Model B')
        plt.ylabel('Model A')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'handshake_convergence_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Convergence distribution
        scores = [r['final_convergence_score'] for r in all_results.values()]
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=self.convergence_threshold, color='red', linestyle='--', 
                   label=f'Convergence threshold ({self.convergence_threshold})')
        plt.axvline(x=np.mean(scores), color='green', linestyle='--', 
                   label=f'Average ({np.mean(scores):.3f})')
        plt.xlabel('Final Convergence Score')
        plt.ylabel('Number of Model Pairs')
        plt.title('Distribution of Handshake Convergence Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'convergence_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Iterations vs Convergence
        iterations = [r['iterations'] for r in all_results.values()]
        plt.figure(figsize=(10, 6))
        plt.scatter(iterations, scores, alpha=0.7, s=60)
        plt.xlabel('Iterations to Completion')
        plt.ylabel('Final Convergence Score')
        plt.title('Iterations vs Final Convergence Score')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(iterations, scores, 1)
        p = np.poly1d(z)
        plt.plot(iterations, p(iterations), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'iterations_vs_convergence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\\n✓ Visualizations saved to {self.results_dir}/")
    
    def print_final_summary(self, results: dict):
        """Print comprehensive final summary"""
        summary = results['summary']
        
        print(f"\\n{'='*80}")
        print(f"HANDSHAKE PROTOCOL RESULTS")
        print(f"{'='*80}")
        print(f"Models tested: {len(self.models)}")
        print(f"Total pairs: {summary['total_pairs']}")
        print(f"Converged pairs: {summary['converged_pairs']}")
        print(f"Convergence rate: {summary['convergence_rate']:.1%}")
        print(f"Average convergence score: {summary['average_convergence_score']:.4f}")
        print(f"Average iterations: {summary['average_iterations']:.1f}")
        
        print(f"\\nBest performing pair:")
        print(f"  {summary['best_pair']['models']}")
        print(f"  Score: {summary['best_pair']['score']:.4f}")
        print(f"  Iterations: {summary['best_pair']['iterations']}")
        
        print(f"\\nWorst performing pair:")
        print(f"  {summary['worst_pair']['models']}")
        print(f"  Score: {summary['worst_pair']['score']:.4f}")
        print(f"  Iterations: {summary['worst_pair']['iterations']}")
        
        # Key insights
        if summary['convergence_rate'] > 0.5:
            print(f"\\n🎉 SUCCESS: Handshake protocol achieved >50% convergence rate!")
        elif summary['average_convergence_score'] > 0.3:
            print(f"\\n📈 PARTIAL SUCCESS: Significant improvement over baseline communication")
        else:
            print(f"\\n❌ LIMITED SUCCESS: Handshake protocol shows minimal improvement")

if __name__ == "__main__":
    print("Starting AI Handshake Protocol Experiment...")
    experiment = HandshakeProtocol()
    results = experiment.run_full_handshake_matrix()