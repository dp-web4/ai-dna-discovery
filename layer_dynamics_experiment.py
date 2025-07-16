#!/usr/bin/env python3
"""
Layer Dynamics Experiment: Testing if model layers change during persistent runtime
Theory: Model layers may exhibit dynamic changes without explicit backpropagation
Focus: deepseek-coder as primary candidate, with controls from other models
"""

import json
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import ollama
import hashlib

class LayerDynamicsExperiment:
    def __init__(self):
        self.models = [
            "deepseek-coder:1.3b",  # Primary candidate
            "phi3:mini",            # Control 1
            "gemma:2b",             # Control 2
            "tinyllama:latest"      # Control 3
        ]
        
        # Test patterns designed to potentially trigger adaptation
        self.test_patterns = {
            "recursive": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "identity": "x = x",
            "paradox": "This statement is false",
            "existence": "∃",
            "loop": "while True: pass",
            "self_reference": "print(print)",
            "metacognitive": "thinking about thinking",
            "null": "",
            "random": "xK9#mP2$vN"
        }
        
        self.results = {
            "timestamps": [],
            "layer_fingerprints": {},
            "activation_changes": {},
            "pattern_responses": {}
        }
        
        # Initialize model states
        for model in self.models:
            self.results["layer_fingerprints"][model] = []
            self.results["activation_changes"][model] = []
            self.results["pattern_responses"][model] = {}
    
    def get_layer_fingerprint(self, model: str, prompt: str) -> Dict:
        """
        Extract a 'fingerprint' of layer activations
        Since we can't directly access internal layers via Ollama,
        we'll use indirect methods to detect changes
        """
        
        # Method 1: Response variability to identical prompts
        responses = []
        embeddings = []
        
        for _ in range(5):
            # Get response
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.0},  # Deterministic
                keep_alive="24h"  # Keep model in GPU memory
            )
            responses.append(response['response'])
            
            # Get embedding
            embed_response = ollama.embeddings(
                model=model,
                prompt=prompt
            )
            embeddings.append(embed_response['embedding'])
        
        # Calculate variability metrics
        response_hashes = [hashlib.md5(r.encode()).hexdigest() for r in responses]
        unique_responses = len(set(response_hashes))
        
        # Embedding variability
        embed_array = np.array(embeddings)
        embed_std = np.std(embed_array, axis=0)
        embed_variability = np.mean(embed_std)
        
        # Method 2: Probe with specific patterns that might reveal internal state
        probe_responses = {}
        for probe_name, probe_pattern in self.test_patterns.items():
            probe_resp = ollama.generate(
                model=model,
                prompt=f"Complete: {probe_pattern}",
                options={"temperature": 0.0, "num_predict": 10},
                keep_alive="24h"
            )
            probe_responses[probe_name] = probe_resp['response']
        
        return {
            "timestamp": datetime.now().isoformat(),
            "unique_responses": unique_responses,
            "embedding_variability": float(embed_variability),
            "probe_responses": probe_responses,
            "response_hashes": response_hashes
        }
    
    def measure_adaptation(self, model: str, training_pattern: str, iterations: int = 50):
        """
        'Train' the model by repeated exposure to a pattern
        Measure if this causes detectable changes in behavior
        """
        
        print(f"\n--- Testing adaptation in {model} ---")
        
        # Baseline fingerprint
        print("Getting baseline fingerprint...")
        baseline = self.get_layer_fingerprint(model, "Hello")
        self.results["layer_fingerprints"][model].append(baseline)
        
        # Repeated exposure phase
        print(f"Exposing to pattern '{training_pattern}' {iterations} times...")
        for i in range(iterations):
            ollama.generate(
                model=model,
                prompt=training_pattern,
                options={"temperature": 0.0},
                keep_alive="24h"
            )
            
            if i % 10 == 0:
                print(f"  Iteration {i}/{iterations}")
        
        # Post-exposure fingerprint
        print("Getting post-exposure fingerprint...")
        post_exposure = self.get_layer_fingerprint(model, "Hello")
        self.results["layer_fingerprints"][model].append(post_exposure)
        
        # Test if the training pattern is now responded to differently
        print("Testing pattern recognition...")
        pattern_tests = {}
        for test_name, test_prompt in {
            "exact": training_pattern,
            "partial": training_pattern[:len(training_pattern)//2],
            "related": f"Similar to: {training_pattern}",
            "unrelated": "The weather is nice today"
        }.items():
            response = ollama.generate(
                model=model,
                prompt=test_prompt,
                options={"temperature": 0.0},
                keep_alive="24h"
            )
            embed = ollama.embeddings(model=model, prompt=test_prompt)
            pattern_tests[test_name] = {
                "response": response['response'][:100],
                "embed_norm": float(np.linalg.norm(embed['embedding'][:100]))
            }
        
        # Calculate changes
        changes = {
            "embedding_variability_delta": post_exposure["embedding_variability"] - baseline["embedding_variability"],
            "response_consistency_delta": post_exposure["unique_responses"] - baseline["unique_responses"],
            "pattern_recognition": pattern_tests
        }
        
        self.results["activation_changes"][model].append(changes)
        
        return changes
    
    def run_persistence_test(self, model: str, duration_minutes: int = 5):
        """
        Test if model behavior changes over time with persistent use
        """
        
        print(f"\n--- Persistence test for {model} ({duration_minutes} minutes) ---")
        
        start_time = time.time()
        fingerprints = []
        
        # Fixed probe to detect changes
        probe = "The meaning of existence is"
        
        while (time.time() - start_time) < (duration_minutes * 60):
            # Get fingerprint
            fingerprint = self.get_layer_fingerprint(model, probe)
            fingerprints.append(fingerprint)
            
            # Active use between measurements
            for _ in range(10):
                prompt = np.random.choice(list(self.test_patterns.values()))
                ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={"temperature": 0.5},
                    keep_alive="24h"
                )
            
            elapsed = (time.time() - start_time) / 60
            print(f"  {elapsed:.1f} minutes elapsed...")
            
            time.sleep(10)  # Measure every 10 seconds
        
        # Analyze temporal changes
        embed_vars = [fp["embedding_variability"] for fp in fingerprints]
        unique_resps = [fp["unique_responses"] for fp in fingerprints]
        
        temporal_analysis = {
            "embedding_variance_trend": np.polyfit(range(len(embed_vars)), embed_vars, 1)[0],
            "response_variance_trend": np.polyfit(range(len(unique_resps)), unique_resps, 1)[0],
            "total_measurements": len(fingerprints),
            "fingerprints": fingerprints
        }
        
        return temporal_analysis
    
    def run_full_experiment(self):
        """
        Run complete layer dynamics experiment
        """
        
        print("Starting Layer Dynamics Experiment")
        print("Theory: Model layers may change during persistent runtime")
        print("="*60)
        
        # Test 1: Adaptation through repetition
        print("\n### TEST 1: ADAPTATION THROUGH REPETITION ###")
        adaptation_results = {}
        
        for model in self.models:
            # Test with recursive pattern (might trigger deeper adaptation)
            changes = self.measure_adaptation(
                model, 
                "def f(x): return f(x-1) if x > 0 else x",
                iterations=100
            )
            adaptation_results[model] = changes
            
            # Cool down period
            time.sleep(5)
        
        # Test 2: Temporal persistence
        print("\n### TEST 2: TEMPORAL PERSISTENCE ###")
        persistence_results = {}
        
        # Focus on deepseek with longer test
        temporal = self.run_persistence_test("deepseek-coder:1.3b", duration_minutes=10)
        persistence_results["deepseek-coder:1.3b"] = temporal
        
        # Quick tests for controls
        for model in ["phi3:mini", "gemma:2b"]:
            temporal = self.run_persistence_test(model, duration_minutes=3)
            persistence_results[model] = temporal
        
        # Analyze results
        self.analyze_results(adaptation_results, persistence_results)
        
        # Save raw data
        with open("layer_dynamics_results.json", "w") as f:
            json.dump({
                "theory": "Layers may exhibit runtime dynamics without backpropagation",
                "timestamp": datetime.now().isoformat(),
                "adaptation_results": adaptation_results,
                "persistence_results": persistence_results,
                "raw_data": self.results
            }, f, indent=2)
        
        print("\n✓ Experiment complete! Results saved to layer_dynamics_results.json")
    
    def analyze_results(self, adaptation_results: Dict, persistence_results: Dict):
        """
        Analyze and visualize findings
        """
        
        print("\n### ANALYSIS ###")
        
        # Check for adaptation evidence
        print("\nAdaptation Evidence:")
        for model, changes in adaptation_results.items():
            delta_embed = changes["embedding_variability_delta"]
            delta_resp = changes["response_consistency_delta"]
            
            print(f"\n{model}:")
            print(f"  Embedding variability change: {delta_embed:+.6f}")
            print(f"  Response consistency change: {delta_resp:+d}")
            
            if abs(delta_embed) > 0.00001 or abs(delta_resp) > 0:
                print(f"  ⚠️  POTENTIAL ADAPTATION DETECTED!")
        
        # Check for temporal dynamics
        print("\n\nTemporal Dynamics:")
        for model, temporal in persistence_results.items():
            embed_trend = temporal["embedding_variance_trend"]
            resp_trend = temporal["response_variance_trend"]
            
            print(f"\n{model}:")
            print(f"  Embedding variance trend: {embed_trend:+.8f}")
            print(f"  Response variance trend: {resp_trend:+.4f}")
            
            if abs(embed_trend) > 0.000001 or abs(resp_trend) > 0.01:
                print(f"  ⚠️  TEMPORAL DYNAMICS DETECTED!")
        
        # Create visualization
        self.create_visualizations(adaptation_results, persistence_results)
    
    def create_visualizations(self, adaptation_results: Dict, persistence_results: Dict):
        """
        Create visual analysis of layer dynamics
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Layer Dynamics Analysis: Testing Runtime Neural Plasticity", fontsize=16)
        
        # Plot 1: Adaptation effects
        ax1 = axes[0, 0]
        models = list(adaptation_results.keys())
        embed_deltas = [adaptation_results[m]["embedding_variability_delta"] for m in models]
        
        bars = ax1.bar(models, embed_deltas, color=['darkred', 'blue', 'green', 'orange'])
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel("Embedding Variability Change")
        ax1.set_title("Adaptation After 100 Iterations")
        ax1.tick_params(axis='x', rotation=45)
        
        # Highlight significant changes
        for i, delta in enumerate(embed_deltas):
            if abs(delta) > 0.00001:
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(3)
        
        # Plot 2: Temporal trends (deepseek focus)
        ax2 = axes[0, 1]
        if "deepseek-coder:1.3b" in persistence_results:
            deepseek_data = persistence_results["deepseek-coder:1.3b"]["fingerprints"]
            times = range(len(deepseek_data))
            embed_vars = [fp["embedding_variability"] for fp in deepseek_data]
            
            ax2.plot(times, embed_vars, 'darkred', linewidth=2, label='deepseek-coder')
            ax2.set_xlabel("Time (measurements)")
            ax2.set_ylabel("Embedding Variability")
            ax2.set_title("Temporal Dynamics: deepseek-coder")
            ax2.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(times, embed_vars, 1)
            p = np.poly1d(z)
            ax2.plot(times, p(times), "r--", alpha=0.8, label=f'Trend: {z[0]:.2e}')
            ax2.legend()
        
        # Plot 3: Response consistency changes
        ax3 = axes[1, 0]
        resp_deltas = [adaptation_results[m]["response_consistency_delta"] for m in models]
        
        bars = ax3.bar(models, resp_deltas, color=['darkred', 'blue', 'green', 'orange'])
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel("Response Consistency Change")
        ax3.set_title("Response Variability After Training")
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Model comparison of temporal trends
        ax4 = axes[1, 1]
        for model, data in persistence_results.items():
            if "fingerprints" in data and len(data["fingerprints"]) > 0:
                embed_vars = [fp["embedding_variability"] for fp in data["fingerprints"]]
                # Normalize to show relative changes
                embed_vars_norm = (embed_vars - embed_vars[0]) / (embed_vars[0] + 1e-10)
                ax4.plot(embed_vars_norm, label=model.split(':')[0], linewidth=2)
        
        ax4.set_xlabel("Time (measurements)")
        ax4.set_ylabel("Normalized Embedding Variability")
        ax4.set_title("Comparative Temporal Dynamics")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("layer_dynamics_analysis.png", dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved to layer_dynamics_analysis.png")

if __name__ == "__main__":
    experiment = LayerDynamicsExperiment()
    experiment.run_full_experiment()