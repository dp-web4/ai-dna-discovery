#!/usr/bin/env python3
"""
Focused AI Handshake Protocol - Constrained to single symbols
"""

import json
import numpy as np
import requests
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

class FocusedHandshakeProtocol:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.chat_url = "http://localhost:11434/api/generate"
        self.results_dir = "handshake_focused_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # All 6 models
        self.models = [
            'phi3:mini', 'gemma:2b', 'tinyllama:latest',
            'qwen2:0.5b', 'deepseek-coder:1.3b', 'llama3.2:1b'
        ]
        
        # Constrained parameters for focused testing
        self.max_iterations = 50
        self.convergence_threshold = 0.7
        self.stability_rounds = 3
        
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
    
    def generate_symbol(self, model: str, prompt: str) -> str:
        """Get single symbol response from model"""
        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": model, 
                    "prompt": prompt + " Respond with ONLY a single symbol, word, or emoji. No explanation.",
                    "stream": False,
                    "options": {"temperature": 0.2, "max_tokens": 10}
                },
                timeout=15
            )
            if response.status_code == 200:
                result = response.json()['response'].strip()
                # Extract first word/symbol only
                return result.split()[0] if result.split() else result
        except:
            pass
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
    
    def run_handshake_pair(self, model_a: str, model_b: str):
        """Run handshake between two models"""
        print(f"\\n--- Handshake: {model_a.split(':')[0]} ↔ {model_b.split(':')[0]} ---")
        
        # Start with existence symbol
        current_symbol = "∃"
        history = []
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"  Round {iteration}: '{current_symbol}'", end=" → ")
            
            # Model A interprets
            prompt_a = f"Given symbol '{current_symbol}', what single symbol captures its essence?"
            response_a = self.generate_symbol(model_a, prompt_a)
            
            # Model B interprets  
            prompt_b = f"Given symbol '{current_symbol}' and interpretation '{response_a}', what single symbol represents this?"
            response_b = self.generate_symbol(model_b, prompt_b)
            
            print(f"{response_a} / {response_b}", end=" ")
            
            # Calculate convergence
            text_match = 1.0 if response_a.lower() == response_b.lower() else 0.0
            
            # Get embeddings for the combined pattern
            combined = f"{response_a}→{response_b}"
            emb_a = self.get_embedding(model_a, combined)
            emb_b = self.get_embedding(model_b, combined)
            emb_similarity = self.calculate_similarity(emb_a, emb_b)
            
            convergence = 0.6 * emb_similarity + 0.4 * text_match
            print(f"(conv: {convergence:.3f})")
            
            # Store history
            history.append({
                'iteration': iteration,
                'input_symbol': current_symbol,
                'response_a': response_a,
                'response_b': response_b,
                'text_match': text_match,
                'embedding_similarity': emb_similarity,
                'convergence_score': convergence
            })
            
            # Check convergence
            if convergence >= self.convergence_threshold:
                print(f"    ✓ CONVERGED at iteration {iteration}")
                return {
                    'model_a': model_a,
                    'model_b': model_b,
                    'converged': True,
                    'iterations': iteration,
                    'final_convergence': convergence,
                    'history': history
                }
            
            # Update symbol for next iteration
            if text_match == 1.0:
                current_symbol = response_a  # They agreed
            else:
                # Combine responses
                current_symbol = f"{response_a}{response_b}"
            
            time.sleep(0.2)
        
        # Did not converge
        final_convergence = history[-1]['convergence_score'] if history else 0.0
        print(f"    ✗ No convergence (final: {final_convergence:.3f})")
        
        return {
            'model_a': model_a,
            'model_b': model_b,
            'converged': False,
            'iterations': self.max_iterations,
            'final_convergence': final_convergence,
            'history': history
        }
    
    def run_full_experiment(self):
        """Run handshakes between all model pairs"""
        print(f"\\n{'='*60}")
        print(f"FOCUSED HANDSHAKE PROTOCOL")
        print(f"{'='*60}")
        print(f"Models: {[m.split(':')[0] for m in self.models]}")
        print(f"Pairs to test: {len(self.models) * (len(self.models) - 1) // 2}")
        
        start_time = datetime.now()
        all_results = []
        
        # Test all unique pairs
        for i, model_a in enumerate(self.models):
            for j, model_b in enumerate(self.models):
                if i < j:
                    try:
                        result = self.run_handshake_pair(model_a, model_b)
                        all_results.append(result)
                    except Exception as e:
                        print(f"    Error: {e}")
                        continue
        
        # Analysis
        converged_count = sum(1 for r in all_results if r['converged'])
        total_pairs = len(all_results)
        convergence_rate = converged_count / total_pairs if total_pairs > 0 else 0
        avg_score = np.mean([r['final_convergence'] for r in all_results])
        avg_iterations = np.mean([r['iterations'] for r in all_results])
        
        print(f"\\n{'='*60}")
        print(f"HANDSHAKE RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total pairs tested: {total_pairs}")
        print(f"Converged pairs: {converged_count}")
        print(f"Convergence rate: {convergence_rate:.1%}")
        print(f"Average final score: {avg_score:.4f}")
        print(f"Average iterations: {avg_iterations:.1f}")
        print(f"Total time: {datetime.now() - start_time}")
        
        # Show best performers
        if all_results:
            best = max(all_results, key=lambda x: x['final_convergence'])
            print(f"\\nBest pair: {best['model_a'].split(':')[0]} ↔ {best['model_b'].split(':')[0]}")
            print(f"  Score: {best['final_convergence']:.4f}")
            print(f"  Converged: {best['converged']}")
            print(f"  Iterations: {best['iterations']}")
        
        # Save results
        final_results = {
            'experiment': 'focused_handshake_protocol',
            'timestamp': datetime.now().isoformat(),
            'models': self.models,
            'parameters': {
                'max_iterations': self.max_iterations,
                'convergence_threshold': self.convergence_threshold
            },
            'results': all_results,
            'summary': {
                'total_pairs': total_pairs,
                'converged_pairs': converged_count,
                'convergence_rate': convergence_rate,
                'average_final_score': avg_score,
                'average_iterations': avg_iterations
            }
        }
        
        # Save to file
        results_file = os.path.join(
            self.results_dir, 
            f"handshake_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Create visualization
        self.create_visualization(all_results)
        
        print(f"\\nResults saved to: {results_file}")
        
        return final_results
    
    def create_visualization(self, results):
        """Create convergence matrix visualization"""
        n_models = len(self.models)
        matrix = np.zeros((n_models, n_models))
        
        # Build convergence matrix
        for result in results:
            model_a = result['model_a']
            model_b = result['model_b']
            score = result['final_convergence']
            
            i = self.models.index(model_a)
            j = self.models.index(model_b)
            
            matrix[i, j] = score
            matrix[j, i] = score  # Symmetric
        
        # Set diagonal to 1.0 (self-similarity)
        np.fill_diagonal(matrix, 1.0)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        model_names = [m.split(':')[0] for m in self.models]
        
        sns.heatmap(matrix, 
                   xticklabels=model_names,
                   yticklabels=model_names,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   center=0.5,
                   vmin=0,
                   vmax=1,
                   square=True)
        
        plt.title('Handshake Protocol Convergence Matrix', fontsize=16)
        plt.xlabel('Model B')
        plt.ylabel('Model A')
        plt.tight_layout()
        
        output_file = os.path.join(self.results_dir, 'handshake_convergence_matrix.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {output_file}")

if __name__ == "__main__":
    experiment = FocusedHandshakeProtocol()
    results = experiment.run_full_experiment()