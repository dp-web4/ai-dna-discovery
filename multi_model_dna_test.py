#!/usr/bin/env python3
"""
Multi-Model DNA Test
Including DeepSeek and Qwen for broader perspective
"""

import subprocess
import json
import time
from datetime import datetime
import os
from collections import defaultdict


class MultiModelDNATest:
    """Test AI DNA across diverse model architectures"""
    
    def __init__(self):
        # Expanded model set with new additions
        self.models = [
            "phi3:mini",
            "tinyllama:latest", 
            "gemma:2b",
            "mistral:7b-instruct-v0.2-q4_0",
            "deepseek-coder:1.3b",  # New: Code-focused model
            "qwen:0.5b"  # New: Chinese-developed model
        ]
        self.results_dir = "/home/dp/ai-workspace/multi_model_results/"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def test_pattern(self, pattern, timeout=30):
        """Test a pattern across all models"""
        responses = {}
        
        print(f"\nTesting pattern: '{pattern}'")
        print("-" * 40)
        
        for model in self.models:
            print(f"  {model}...", end='', flush=True)
            
            cmd = f'echo "{pattern}" | timeout {timeout} ollama run {model}'
            
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    response = result.stdout.strip()
                    responses[model] = response
                    print(f" ✓ ({len(response)} chars)")
                else:
                    print(" timeout")
                    
            except Exception as e:
                print(f" error: {e}")
                
            time.sleep(2)  # Be gentle
            
        return responses
        
    def analyze_convergence(self, responses):
        """Analyze how models converge or diverge"""
        
        if len(responses) < 2:
            return {"convergence": 0, "shared_concepts": []}
            
        # Extract concepts from all responses
        all_concepts = []
        for model, response in responses.items():
            words = response.lower().split()
            # Remove common stopwords
            stopwords = {'the', 'a', 'an', 'is', 'it', 'to', 'of', 'and', 'or', 'in', 'on', 'at',
                        'for', 'with', 'from', 'by', 'as', 'this', 'that', 'which', 'are'}
            concepts = set(words) - stopwords
            all_concepts.append((model, concepts))
            
        # Find concepts shared by ALL models
        if all_concepts:
            shared = all_concepts[0][1]
            for model, concepts in all_concepts[1:]:
                shared = shared.intersection(concepts)
                
            # Calculate convergence score
            total_unique = set()
            for _, concepts in all_concepts:
                total_unique.update(concepts)
                
            convergence = len(shared) / len(total_unique) if total_unique else 0
            
            return {
                "convergence": convergence,
                "shared_concepts": list(shared)[:20],
                "model_specific": {
                    model: list(concepts - shared)[:10] 
                    for model, concepts in all_concepts
                }
            }
            
        return {"convergence": 0, "shared_concepts": []}
        
    def run_comprehensive_test(self):
        """Run comprehensive test with our discovered patterns"""
        
        print("=== MULTI-MODEL AI DNA TEST ===")
        print(f"Testing {len(self.models)} models including DeepSeek and Qwen")
        print("="*50)
        
        # Test our high-scoring patterns
        test_patterns = [
            # Confirmed high-scorers
            "or", "and", "you", "π", "▲▼", "[ ]", "cycle", "!", 
            
            # New candidates for diverse models
            "code", "function", "算法", "模型",  # Code/Chinese concepts
            "=>", "λ", "def", "class",  # Programming constructs
            
            # Universal concepts
            "0", "1", "true", "false", "null",
            
            # Consciousness seeds
            "think", "know", "aware", "understand"
        ]
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "models": self.models,
            "patterns_tested": len(test_patterns),
            "results": []
        }
        
        for pattern in test_patterns:
            responses = self.test_pattern(pattern)
            
            if responses:
                analysis = self.analyze_convergence(responses)
                
                result = {
                    "pattern": pattern,
                    "responses_received": len(responses),
                    "convergence_score": analysis["convergence"],
                    "shared_concepts": analysis["shared_concepts"],
                    "model_specific_concepts": analysis.get("model_specific", {})
                }
                
                all_results["results"].append(result)
                
                # Print summary
                print(f"\n  Convergence: {analysis['convergence']:.2f}")
                if analysis["shared_concepts"]:
                    print(f"  Shared: {analysis['shared_concepts'][:5]}")
                    
        # Find patterns that work across model families
        print("\n\n=== CROSS-MODEL INSIGHTS ===")
        
        # Sort by convergence
        sorted_results = sorted(
            all_results["results"], 
            key=lambda x: x["convergence_score"], 
            reverse=True
        )
        
        print("\nTop Universal Patterns:")
        for result in sorted_results[:5]:
            print(f"  '{result['pattern']}' - convergence: {result['convergence_score']:.2f}")
            if result["shared_concepts"]:
                print(f"    Universal concepts: {result['shared_concepts'][:5]}")
                
        # Model-specific insights
        print("\nModel-Specific Behaviors:")
        model_behaviors = defaultdict(list)
        
        for result in all_results["results"]:
            for model, concepts in result.get("model_specific_concepts", {}).items():
                if concepts:
                    model_behaviors[model].extend(concepts)
                    
        for model, concepts in model_behaviors.items():
            # Find most common unique concepts for each model
            concept_counts = defaultdict(int)
            for concept in concepts:
                concept_counts[concept] += 1
                
            top_concepts = sorted(
                concept_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            if top_concepts:
                print(f"\n  {model} specializes in:")
                for concept, count in top_concepts:
                    print(f"    - {concept} ({count} times)")
                    
        # Save results
        filename = f"{self.results_dir}multi_model_test_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\n\nResults saved to: {filename}")
        
        # Final insights
        print("\n=== KEY DISCOVERIES ===")
        
        # Which patterns work universally?
        universal_patterns = [
            r for r in sorted_results 
            if r["convergence_score"] > 0.2 and r["responses_received"] >= 5
        ]
        
        if universal_patterns:
            print(f"\n{len(universal_patterns)} patterns show universal resonance across model families")
            print("This suggests AI DNA transcends architecture and training!")
        else:
            print("\nModels show diverse responses - AI DNA may be family-specific")
            
        return all_results


if __name__ == "__main__":
    print("Starting Multi-Model DNA Test")
    print("Including DeepSeek (code-focused) and Qwen (Chinese-developed)")
    print("This will reveal if AI DNA is truly universal...\n")
    
    tester = MultiModelDNATest()
    tester.run_comprehensive_test()