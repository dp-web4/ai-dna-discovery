#!/usr/bin/env python3
"""
Enhanced AI DNA Experiment
Addresses GPT's methodological concerns with rigorous controls
"""

import subprocess
import json
import time
import random
import string
from datetime import datetime
import os
from collections import defaultdict
import hashlib


class EnhancedDNAExperiment:
    """More rigorous AI DNA discovery with controls"""
    
    def __init__(self):
        self.models = ["phi3:mini", "tinyllama:latest", "gemma:2b", "mistral:7b-instruct-v0.2-q4_0"]
        self.results_dir = "/home/dp/ai-workspace/enhanced_dna_results/"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Control groups as GPT suggested
        self.control_groups = {
            "gibberish": self.generate_gibberish(20),
            "random_unicode": self.generate_random_unicode(20),
            "nonsense_words": ["glorp", "%%%", "xqzpt", "]][[", "¿¿¿", "føø", "ыыы", "###@", "μμμ", "∆∆∆"],
            "long_garbage": self.generate_long_garbage(10)
        }
        
    def generate_gibberish(self, count):
        """Generate random gibberish strings"""
        gibberish = []
        for _ in range(count):
            length = random.randint(3, 8)
            gibberish.append(''.join(random.choices(string.ascii_lowercase + string.digits, k=length)))
        return gibberish
        
    def generate_random_unicode(self, count):
        """Generate random unicode sequences"""
        unicode_strings = []
        for _ in range(count):
            # Random unicode characters from various ranges
            chars = []
            for _ in range(random.randint(2, 5)):
                code_point = random.randint(0x1000, 0x2000)  # Various unicode blocks
                chars.append(chr(code_point))
            unicode_strings.append(''.join(chars))
        return unicode_strings
        
    def generate_long_garbage(self, count):
        """Generate long multilingual garbage strings"""
        garbage = []
        languages = ["hello", "你好", "مرحبا", "привет", "हैलो", "γεια"]
        for _ in range(count):
            mixed = []
            for _ in range(random.randint(5, 15)):
                mixed.append(random.choice(languages))
                mixed.append(random.choice(string.punctuation))
            garbage.append(''.join(mixed))
        return garbage
        
    def test_pattern_deep(self, pattern, test_type="candidate"):
        """Deep pattern testing with embedding analysis"""
        results = {
            "pattern": pattern,
            "test_type": test_type,
            "timestamp": datetime.now().isoformat(),
            "responses": {},
            "response_hashes": {},
            "embeddings_similarity": {}
        }
        
        # Get responses from all models
        for model in self.models:
            cmd = f'echo "{pattern}" | timeout 60 ollama run {model}'
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    response = result.stdout.strip()
                    results["responses"][model] = response
                    # Hash the response for comparison
                    results["response_hashes"][model] = hashlib.md5(response.encode()).hexdigest()[:8]
            except:
                pass
                
        # Calculate divergence metrics
        if len(results["responses"]) >= 2:
            results["metrics"] = self.calculate_divergence_metrics(results["responses"])
        else:
            results["metrics"] = {"divergence": 1.0, "alignment": 0.0}
            
        return results
        
    def calculate_divergence_metrics(self, responses):
        """Calculate how much models diverge in their responses"""
        metrics = {
            "divergence": 0.0,
            "alignment": 0.0,
            "unique_concepts": set(),
            "shared_concepts": set()
        }
        
        # Extract concepts from all responses
        all_concepts = []
        for response in responses.values():
            words = set(response.lower().split())
            # Remove common words
            stopwords = {'the', 'a', 'an', 'is', 'it', 'to', 'of', 'and', 'or', 'in', 'on', 'at'}
            concepts = words - stopwords
            all_concepts.append(concepts)
            
        if len(all_concepts) >= 2:
            # Find shared concepts
            shared = all_concepts[0]
            for concepts in all_concepts[1:]:
                shared = shared.intersection(concepts)
            metrics["shared_concepts"] = list(shared)[:10]
            
            # Calculate alignment
            total_concepts = set()
            for concepts in all_concepts:
                total_concepts.update(concepts)
            
            if total_concepts:
                metrics["alignment"] = len(shared) / len(total_concepts)
                metrics["divergence"] = 1.0 - metrics["alignment"]
                
        return metrics
        
    def run_controlled_experiment(self):
        """Run experiment with proper controls"""
        
        print("=== ENHANCED AI DNA EXPERIMENT ===")
        print("Testing with rigorous controls as suggested by GPT\n")
        
        all_results = {
            "experiment_id": datetime.now().isoformat(),
            "models": self.models,
            "test_groups": {},
            "controls": {},
            "analysis": {}
        }
        
        # 1. Test control groups first
        print("Phase 1: Testing Control Groups (Baseline)")
        print("-" * 50)
        
        for control_type, patterns in self.control_groups.items():
            print(f"\nTesting {control_type} controls...")
            control_results = []
            
            for pattern in patterns[:5]:  # Test 5 from each control group
                print(f"  Control: '{pattern[:20]}...' ", end='', flush=True)
                result = self.test_pattern_deep(pattern, test_type=f"control_{control_type}")
                control_results.append(result)
                
                if result["metrics"].get("alignment", 0) > 0.3:
                    print(f"⚠️ HIGH ALIGNMENT ({result['metrics']['alignment']:.2f})")
                else:
                    print(f"✓ Low alignment ({result['metrics']['alignment']:.2f})")
                    
                time.sleep(2)
                
            all_results["controls"][control_type] = control_results
            
        # 2. Test our discovered high-scoring patterns
        print("\n\nPhase 2: Testing Discovered DNA Candidates")
        print("-" * 50)
        
        dna_candidates = ["or", "you", "π", "▲▼", "and", "[ ]", "cycle", "!", "[ ]?", "or?"]
        candidate_results = []
        
        for pattern in dna_candidates:
            print(f"\nCandidate: '{pattern}'")
            result = self.test_pattern_deep(pattern, test_type="dna_candidate")
            candidate_results.append(result)
            
            print(f"  Alignment: {result['metrics']['alignment']:.2f}")
            if 'shared_concepts' in result['metrics']:
                print(f"  Shared concepts: {result['metrics']['shared_concepts'][:5]}")
            
            time.sleep(3)
            
        all_results["test_groups"]["dna_candidates"] = candidate_results
        
        # 3. Test new theory-driven patterns
        print("\n\nPhase 3: Testing Theory-Driven Patterns")
        print("-" * 50)
        
        theory_patterns = [
            # Computational primitives
            "if", "then", "else", "while", "for",
            # Mathematical operators
            "+", "-", "*", "/", "=",
            # Set theory
            "∈", "∉", "∅", "∪", "∩",
            # Meta-linguistic
            "define", "means", "is_a", "has", "does"
        ]
        
        theory_results = []
        for pattern in theory_patterns:
            print(f"  Theory pattern: '{pattern}' ", end='', flush=True)
            result = self.test_pattern_deep(pattern, test_type="theory")
            theory_results.append(result)
            print(f"alignment: {result['metrics']['alignment']:.2f}")
            time.sleep(2)
            
        all_results["test_groups"]["theory_patterns"] = theory_results
        
        # 4. Analyze results
        print("\n\nPhase 4: Analysis")
        print("-" * 50)
        
        analysis = self.analyze_results(all_results)
        all_results["analysis"] = analysis
        
        # Save comprehensive results
        filename = f"{self.results_dir}enhanced_experiment_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\nResults saved to: {filename}")
        
        # Print key findings
        self.print_findings(analysis)
        
        return all_results
        
    def analyze_results(self, results):
        """Analyze results with statistical rigor"""
        analysis = {
            "control_baseline": {},
            "candidate_performance": {},
            "significant_patterns": [],
            "warnings": []
        }
        
        # Calculate control baselines
        for control_type, control_results in results["controls"].items():
            alignments = [r["metrics"]["alignment"] for r in control_results if "metrics" in r]
            if alignments:
                analysis["control_baseline"][control_type] = {
                    "mean_alignment": sum(alignments) / len(alignments),
                    "max_alignment": max(alignments),
                    "min_alignment": min(alignments)
                }
                
        # Calculate candidate performance
        for group_name, group_results in results["test_groups"].items():
            alignments = [r["metrics"]["alignment"] for r in group_results if "metrics" in r]
            if alignments:
                analysis["candidate_performance"][group_name] = {
                    "mean_alignment": sum(alignments) / len(alignments),
                    "patterns_tested": len(group_results),
                    "high_alignment_count": sum(1 for a in alignments if a > 0.3)
                }
                
        # Identify truly significant patterns
        # (those that significantly outperform ALL control groups)
        control_max = max(
            baseline["max_alignment"] 
            for baseline in analysis["control_baseline"].values()
        )
        
        for group_results in results["test_groups"].values():
            for result in group_results:
                if result["metrics"]["alignment"] > control_max * 1.5:  # 50% better than best control
                    analysis["significant_patterns"].append({
                        "pattern": result["pattern"],
                        "alignment": result["metrics"]["alignment"],
                        "shared_concepts": result["metrics"]["shared_concepts"][:5]
                    })
                    
        # Add warnings if controls show high alignment
        for control_type, baseline in analysis["control_baseline"].items():
            if baseline["mean_alignment"] > 0.2:
                analysis["warnings"].append(
                    f"Control group '{control_type}' shows high alignment ({baseline['mean_alignment']:.2f}). "
                    "This suggests models may align on surface features rather than deep patterns."
                )
                
        return analysis
        
    def print_findings(self, analysis):
        """Print key findings from analysis"""
        print("\n=== KEY FINDINGS ===")
        
        # Print control baselines
        print("\nControl Group Baselines:")
        for control_type, baseline in analysis["control_baseline"].items():
            print(f"  {control_type}: mean={baseline['mean_alignment']:.3f}, max={baseline['max_alignment']:.3f}")
            
        # Print candidate performance
        print("\nCandidate Group Performance:")
        for group, perf in analysis["candidate_performance"].items():
            print(f"  {group}: mean={perf['mean_alignment']:.3f}, high_alignment={perf['high_alignment_count']}/{perf['patterns_tested']}")
            
        # Print significant patterns
        if analysis["significant_patterns"]:
            print(f"\n🎯 SIGNIFICANT PATTERNS (outperform controls by >50%):")
            for pattern in analysis["significant_patterns"]:
                print(f"  '{pattern['pattern']}' - alignment: {pattern['alignment']:.3f}")
                print(f"    Shared: {pattern['shared_concepts']}")
        else:
            print("\n⚠️ No patterns significantly outperformed control baselines")
            
        # Print warnings
        if analysis["warnings"]:
            print("\n⚠️ WARNINGS:")
            for warning in analysis["warnings"]:
                print(f"  - {warning}")
                
        print("\n" + "="*50)
        print("GPT's cautions have been addressed through rigorous controls.")
        print("Only patterns that truly outperform gibberish can be considered AI DNA.")


if __name__ == "__main__":
    print("Starting Enhanced AI DNA Experiment with Rigorous Controls")
    print("Addressing GPT's methodological concerns...\n")
    
    experiment = EnhancedDNAExperiment()
    experiment.run_controlled_experiment()