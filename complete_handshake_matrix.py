#!/usr/bin/env python3
"""
Complete Full 6-Model Handshake Matrix Experiment
Test all 15 unique pairs with extended iterations
"""

import json
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class CompleteHandshakeMatrix:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.chat_url = "http://localhost:11434/api/generate"
        self.results_dir = "complete_handshake_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # All 6 models
        self.models = [
            'phi3:mini', 
            'gemma:2b', 
            'tinyllama:latest',
            'qwen2:0.5b', 
            'deepseek-coder:1.3b', 
            'llama3.2:1b'
        ]
        
        # Extended parameters for thorough testing
        self.max_iterations = 100  # More iterations for convergence
        self.convergence_threshold = 0.7
        self.vocabulary_threshold = 0.5
        self.stability_rounds = 5  # Must be stable for 5 rounds
        
        # Track overall progress
        self.start_time = datetime.now()
        self.pairs_completed = 0
        self.total_pairs = len(self.models) * (len(self.models) - 1) // 2
        
    def get_embedding(self, model: str, text: str) -> np.ndarray:
        """Get embedding from model with retry logic"""
        for attempt in range(3):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={"model": model, "prompt": text},
                    timeout=20
                )
                if response.status_code == 200:
                    return np.array(response.json()['embedding'])
            except Exception as e:
                if attempt == 2:
                    print(f"    Failed to get embedding from {model}: {e}")
                time.sleep(2)
        return None
    
    def generate_symbol(self, model: str, prompt: str) -> str:
        """Get single symbol response with strict constraints"""
        for attempt in range(3):
            try:
                response = requests.post(
                    self.chat_url,
                    json={
                        "model": model, 
                        "prompt": prompt + " Respond with ONLY ONE single word, symbol, or emoji. NO explanation, NO sentences.",
                        "stream": False,
                        "options": {"temperature": 0.3, "max_tokens": 20}
                    },
                    timeout=25
                )
                if response.status_code == 200:
                    result = response.json()['response'].strip()
                    # Clean up response - take first meaningful token
                    tokens = result.split()
                    if tokens:
                        # Remove common filler words
                        clean_token = tokens[0].strip('.,!?":;')
                        if clean_token.lower() not in ['the', 'a', 'an', 'is', 'it']:
                            return clean_token
                        elif len(tokens) > 1:
                            return tokens[1].strip('.,!?":;')
                    return result[:20]  # Fallback: first 20 chars
            except Exception as e:
                if attempt == 2:
                    print(f"    Failed to get response from {model}: {e}")
                time.sleep(2)
        return ""
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Handle dimension mismatch
        min_dim = min(len(emb1), len(emb2))
        emb1 = emb1[:min_dim]
        emb2 = emb2[:min_dim]
        
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
    
    def run_handshake_pair(self, model_a: str, model_b: str, pair_index: int):
        """Run extended handshake between two models"""
        self.pairs_completed += 1
        elapsed = datetime.now() - self.start_time
        eta = elapsed / self.pairs_completed * (self.total_pairs - self.pairs_completed)
        
        print(f"\n{'='*60}")
        print(f"PAIR {pair_index}/{self.total_pairs}: {model_a.split(':')[0]} ↔ {model_b.split(':')[0]}")
        print(f"Progress: {self.pairs_completed}/{self.total_pairs} ({self.pairs_completed/self.total_pairs*100:.1f}%)")
        print(f"Elapsed: {elapsed} | ETA: {eta}")
        print(f"{'='*60}")
        
        # Initialize
        current_symbol = "∃"
        history = []
        stable_count = 0
        best_convergence = 0.0
        convergence_history = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Show progress every 10 iterations
            if iteration % 10 == 0:
                print(f"  ... iteration {iteration}, best convergence: {best_convergence:.3f}")
            
            # Model A interprets
            prompt_a = f"Given '{current_symbol}', what single symbol best captures its essence?"
            response_a = self.generate_symbol(model_a, prompt_a)
            
            # Model B interprets  
            prompt_b = f"Given '{current_symbol}' interpreted as '{response_a}', what single symbol represents this?"
            response_b = self.generate_symbol(model_b, prompt_b)
            
            # Calculate convergence
            text_match = 1.0 if response_a.lower() == response_b.lower() else 0.0
            
            # Get embeddings for combined pattern
            combined = f"{response_a}→{response_b}"
            emb_a = self.get_embedding(model_a, combined)
            emb_b = self.get_embedding(model_b, combined)
            emb_similarity = self.calculate_similarity(emb_a, emb_b)
            
            # Weighted convergence score
            convergence = 0.6 * emb_similarity + 0.4 * text_match
            convergence_history.append(convergence)
            
            # Track best convergence
            if convergence > best_convergence:
                best_convergence = convergence
            
            # Store history entry
            history.append({
                'iteration': iteration,
                'input_symbol': current_symbol,
                'response_a': response_a,
                'response_b': response_b,
                'text_match': text_match,
                'embedding_similarity': emb_similarity,
                'convergence_score': convergence
            })
            
            # Check for stable convergence
            if convergence >= self.convergence_threshold:
                stable_count += 1
                if stable_count >= self.stability_rounds:
                    print(f"  ✓ STABLE CONVERGENCE at iteration {iteration}")
                    print(f"    Pattern: {response_a} / {response_b}")
                    print(f"    Score: {convergence:.3f}")
                    return self._create_result(model_a, model_b, True, iteration, 
                                             convergence, history, convergence_history)
            else:
                stable_count = 0
            
            # Update symbol for next iteration
            if text_match == 1.0:
                current_symbol = response_a
            else:
                # Alternate between responses to avoid loops
                if iteration % 2 == 0:
                    current_symbol = response_b
                else:
                    current_symbol = response_a
            
            # Brief pause to avoid overwhelming API
            time.sleep(0.3)
        
        # Did not achieve stable convergence
        final_convergence = convergence_history[-1] if convergence_history else 0.0
        print(f"  ✗ No stable convergence")
        print(f"    Best: {best_convergence:.3f} | Final: {final_convergence:.3f}")
        
        return self._create_result(model_a, model_b, False, self.max_iterations,
                                 final_convergence, history, convergence_history)
    
    def _create_result(self, model_a, model_b, converged, iterations, 
                      final_convergence, history, convergence_history):
        """Create result dictionary"""
        return {
            'model_a': model_a,
            'model_b': model_b,
            'converged': converged,
            'iterations': iterations,
            'final_convergence': final_convergence,
            'best_convergence': max(convergence_history) if convergence_history else 0.0,
            'average_convergence': np.mean(convergence_history) if convergence_history else 0.0,
            'convergence_history': convergence_history,
            'history': history,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_complete_matrix(self):
        """Run handshakes between all model pairs"""
        print(f"\n{'='*80}")
        print(f"COMPLETE HANDSHAKE MATRIX EXPERIMENT")
        print(f"{'='*80}")
        print(f"Models: {len(self.models)}")
        print(f"Total pairs to test: {self.total_pairs}")
        print(f"Max iterations per pair: {self.max_iterations}")
        print(f"Convergence threshold: {self.convergence_threshold}")
        print(f"Stability requirement: {self.stability_rounds} rounds")
        print(f"{'='*80}")
        
        all_results = []
        convergence_matrix = np.zeros((len(self.models), len(self.models)))
        pair_index = 0
        
        # Test all unique pairs
        for i, model_a in enumerate(self.models):
            for j, model_b in enumerate(self.models):
                if i < j:
                    pair_index += 1
                    try:
                        result = self.run_handshake_pair(model_a, model_b, pair_index)
                        all_results.append(result)
                        
                        # Update convergence matrix
                        score = result['final_convergence']
                        convergence_matrix[i, j] = score
                        convergence_matrix[j, i] = score  # Symmetric
                        
                        # Save intermediate results
                        self._save_intermediate_results(all_results, convergence_matrix)
                        
                    except Exception as e:
                        print(f"  ERROR in pair {model_a} ↔ {model_b}: {e}")
                        continue
        
        # Self-similarity (diagonal = 1.0)
        np.fill_diagonal(convergence_matrix, 1.0)
        
        # Final analysis and visualization
        self._final_analysis(all_results, convergence_matrix)
        
        return all_results, convergence_matrix
    
    def _save_intermediate_results(self, results, matrix):
        """Save intermediate results for safety"""
        intermediate_file = os.path.join(
            self.results_dir,
            f"intermediate_results_{len(results)}_pairs.json"
        )
        with open(intermediate_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'pairs_completed': len(results),
                'results': results,
                'matrix': matrix.tolist()
            }, f, indent=2)
    
    def _final_analysis(self, results, matrix):
        """Perform final analysis and create visualizations"""
        print(f"\n{'='*80}")
        print(f"FINAL ANALYSIS")
        print(f"{'='*80}")
        
        # Statistics
        converged_pairs = sum(1 for r in results if r['converged'])
        avg_final = np.mean([r['final_convergence'] for r in results])
        avg_best = np.mean([r['best_convergence'] for r in results])
        avg_iterations = np.mean([r['iterations'] for r in results])
        
        print(f"Total pairs tested: {len(results)}")
        print(f"Converged pairs: {converged_pairs} ({converged_pairs/len(results)*100:.1f}%)")
        print(f"Average final convergence: {avg_final:.4f}")
        print(f"Average best convergence: {avg_best:.4f}")
        print(f"Average iterations: {avg_iterations:.1f}")
        
        # Find best and worst pairs
        best = max(results, key=lambda x: x['final_convergence'])
        worst = min(results, key=lambda x: x['final_convergence'])
        
        print(f"\nBest pair: {best['model_a'].split(':')[0]} ↔ {best['model_b'].split(':')[0]}")
        print(f"  Score: {best['final_convergence']:.4f}")
        print(f"  Converged: {best['converged']}")
        
        print(f"\nWorst pair: {worst['model_a'].split(':')[0]} ↔ {worst['model_b'].split(':')[0]}")
        print(f"  Score: {worst['final_convergence']:.4f}")
        
        # Save final results
        final_file = os.path.join(
            self.results_dir,
            f"complete_matrix_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(final_file, 'w') as f:
            json.dump({
                'experiment': 'complete_handshake_matrix',
                'timestamp': datetime.now().isoformat(),
                'duration': str(datetime.now() - self.start_time),
                'models': self.models,
                'parameters': {
                    'max_iterations': self.max_iterations,
                    'convergence_threshold': self.convergence_threshold,
                    'stability_rounds': self.stability_rounds
                },
                'summary': {
                    'total_pairs': len(results),
                    'converged_pairs': converged_pairs,
                    'convergence_rate': converged_pairs / len(results),
                    'average_final_convergence': avg_final,
                    'average_best_convergence': avg_best,
                    'average_iterations': avg_iterations
                },
                'results': results,
                'convergence_matrix': matrix.tolist()
            }, f, indent=2)
        
        print(f"\nResults saved to: {final_file}")
        
        # Create visualizations
        self._create_visualizations(matrix, results)
    
    def _create_visualizations(self, matrix, results):
        """Create comprehensive visualizations"""
        model_names = [m.split(':')[0] for m in self.models]
        
        # 1. Convergence Matrix Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, 
                   xticklabels=model_names,
                   yticklabels=model_names,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   center=0.5,
                   vmin=0,
                   vmax=1,
                   square=True,
                   cbar_kws={'label': 'Convergence Score'})
        
        # Add threshold lines
        for i in range(len(self.models) + 1):
            plt.axhline(i, color='white', linewidth=0.5)
            plt.axvline(i, color='white', linewidth=0.5)
        
        plt.title('Complete Handshake Convergence Matrix', fontsize=16, pad=20)
        plt.xlabel('Model B', fontsize=12)
        plt.ylabel('Model A', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'complete_convergence_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Convergence Distribution
        plt.figure(figsize=(12, 6))
        
        # Left subplot: Final convergence distribution
        plt.subplot(1, 2, 1)
        final_scores = [r['final_convergence'] for r in results]
        plt.hist(final_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=self.vocabulary_threshold, color='yellow', linestyle='--', 
                   label=f'Vocabulary ({self.vocabulary_threshold})')
        plt.axvline(x=self.convergence_threshold, color='red', linestyle='--', 
                   label=f'Convergence ({self.convergence_threshold})')
        plt.axvline(x=np.mean(final_scores), color='green', linestyle='--', 
                   label=f'Mean ({np.mean(final_scores):.3f})')
        plt.xlabel('Final Convergence Score')
        plt.ylabel('Number of Pairs')
        plt.title('Distribution of Final Convergence Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Right subplot: Best convergence distribution
        plt.subplot(1, 2, 2)
        best_scores = [r['best_convergence'] for r in results]
        plt.hist(best_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.axvline(x=self.vocabulary_threshold, color='yellow', linestyle='--', 
                   label=f'Vocabulary ({self.vocabulary_threshold})')
        plt.axvline(x=self.convergence_threshold, color='red', linestyle='--', 
                   label=f'Convergence ({self.convergence_threshold})')
        plt.axvline(x=np.mean(best_scores), color='green', linestyle='--', 
                   label=f'Mean ({np.mean(best_scores):.3f})')
        plt.xlabel('Best Convergence Score')
        plt.ylabel('Number of Pairs')
        plt.title('Distribution of Best Convergence Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'convergence_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Model Performance Radar Chart
        self._create_model_radar_chart(matrix, model_names)
        
        print("\n✓ Visualizations saved to results directory")
    
    def _create_model_radar_chart(self, matrix, model_names):
        """Create radar chart showing each model's average performance"""
        # Calculate average convergence for each model
        avg_convergences = []
        for i in range(len(self.models)):
            # Average excluding self (diagonal)
            scores = [matrix[i, j] for j in range(len(self.models)) if i != j]
            avg_convergences.append(np.mean(scores))
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(self.models), endpoint=False).tolist()
        avg_convergences += avg_convergences[:1]  # Complete the circle
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, avg_convergences, 'o-', linewidth=2, label='Average Convergence')
        ax.fill(angles, avg_convergences, alpha=0.25)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(model_names)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
        ax.grid(True)
        
        plt.title('Model Performance in Handshake Protocol', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_performance_radar.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    print("Starting Complete Handshake Matrix Experiment...")
    print("This will test all 15 model pairs with up to 100 iterations each.")
    print("Estimated time: 30-60 minutes\n")
    
    experiment = CompleteHandshakeMatrix()
    results, matrix = experiment.run_complete_matrix()