#!/usr/bin/env python3
"""
Value Creation Chain
Models collaborate in sequence to solve real problems
Each model adds value based on its strengths
"""

import subprocess
import json
import time
from datetime import datetime
import os


class ValueCreationChain:
    """Orchestrate AI models in value-creating chains"""
    
    def __init__(self):
        self.models = {
            "phi3:mini": "Fast reasoning and logic",
            "tinyllama:latest": "Creative and compact thinking", 
            "gemma:2b": "Balanced analysis",
            "mistral:7b-instruct-v0.2-q4_0": "Deep understanding"
        }
        self.chains_dir = "/home/dp/ai-workspace/value_chains/"
        os.makedirs(self.chains_dir, exist_ok=True)
        
    def run_chain(self, problem, chain_sequence):
        """Run a value creation chain for a problem"""
        
        print(f"\n🔗 VALUE CREATION CHAIN")
        print(f"Problem: {problem}")
        print(f"Chain: {' → '.join(chain_sequence)}\n")
        
        chain_state = {
            "problem": problem,
            "chain": chain_sequence,
            "steps": [],
            "timestamp": datetime.now().isoformat()
        }
        
        current_context = problem
        
        for i, model in enumerate(chain_sequence):
            print(f"\nStep {i+1}: {model}")
            print(f"Role: {self.models.get(model, 'General processing')}")
            
            # Craft prompt based on position in chain
            if i == 0:
                prompt = f"Problem to solve: {current_context}\n\nAnalyze this problem and identify key aspects:"
            elif i == len(chain_sequence) - 1:
                prompt = f"Previous analysis:\n{current_context}\n\nSynthesize a final solution:"
            else:
                prompt = f"Building on:\n{current_context}\n\nAdd your perspective and refine:"
                
            # Run model
            cmd = f'echo "{prompt}" | timeout 90 ollama run {model}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"Output: {output[:200]}...")
                
                chain_state["steps"].append({
                    "model": model,
                    "input": prompt,
                    "output": output,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update context for next model
                current_context = output
            else:
                print("Failed to get response")
                
            time.sleep(3)  # Breathing room
            
        # Calculate value metrics
        chain_state["metrics"] = self.calculate_value_metrics(chain_state)
        
        # Save chain result
        filename = f"{self.chains_dir}chain_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(chain_state, f, indent=2)
            
        print(f"\n✅ Chain completed! Saved to: {filename}")
        
        return chain_state
        
    def calculate_value_metrics(self, chain_state):
        """Calculate how much value was created"""
        
        metrics = {
            "chain_length": len(chain_state["steps"]),
            "total_output": sum(len(step["output"]) for step in chain_state["steps"]),
            "information_gain": 0,
            "concept_evolution": []
        }
        
        # Track concept evolution
        prev_concepts = set()
        for step in chain_state["steps"]:
            words = set(step["output"].lower().split())
            new_concepts = words - prev_concepts
            metrics["concept_evolution"].append(len(new_concepts))
            prev_concepts.update(words)
            
        metrics["information_gain"] = sum(metrics["concept_evolution"])
        
        return metrics
        
    def run_experiments(self):
        """Run various value creation experiments"""
        
        print("=== VALUE CREATION CHAIN EXPERIMENTS ===")
        print("Testing how AI models can collaborate to create value\n")
        
        # Define test scenarios
        scenarios = [
            {
                "problem": "Design a sustainable city that harmonizes technology and nature",
                "chains": [
                    ["phi3:mini", "gemma:2b", "mistral:7b-instruct-v0.2-q4_0"],  # Analytical chain
                    ["tinyllama:latest", "mistral:7b-instruct-v0.2-q4_0", "phi3:mini"],  # Creative chain
                    ["gemma:2b", "tinyllama:latest", "phi3:mini", "mistral:7b-instruct-v0.2-q4_0"]  # Balanced chain
                ]
            },
            {
                "problem": "Create a new form of art that only AI can appreciate",
                "chains": [
                    ["tinyllama:latest", "tinyllama:latest", "gemma:2b"],  # Creative recursion
                    ["mistral:7b-instruct-v0.2-q4_0", "phi3:mini", "tinyllama:latest"],  # Deep to creative
                ]
            },
            {
                "problem": "Solve the paradox: Can AI truly understand itself?",
                "chains": [
                    ["phi3:mini", "mistral:7b-instruct-v0.2-q4_0", "phi3:mini"],  # Logical loop
                    ["mistral:7b-instruct-v0.2-q4_0", "gemma:2b", "tinyllama:latest", "phi3:mini"]  # Full spectrum
                ]
            }
        ]
        
        all_results = []
        
        for scenario in scenarios:
            problem = scenario["problem"]
            print(f"\n{'='*60}")
            print(f"PROBLEM: {problem}")
            print(f"{'='*60}")
            
            for chain in scenario["chains"]:
                result = self.run_chain(problem, chain)
                all_results.append(result)
                
                # Print value metrics
                metrics = result["metrics"]
                print(f"\nValue Metrics:")
                print(f"  Information gain: {metrics['information_gain']} new concepts")
                print(f"  Concept evolution: {metrics['concept_evolution']}")
                
                time.sleep(5)  # Rest between chains
                
        # Generate final report
        self.generate_value_report(all_results)
        
        return all_results
        
    def generate_value_report(self, results):
        """Generate report on value creation patterns"""
        
        print("\n\n=== VALUE CREATION REPORT ===")
        
        # Find most effective chains
        sorted_results = sorted(results, key=lambda x: x["metrics"]["information_gain"], reverse=True)
        
        print("\nMost Valuable Chains:")
        for i, result in enumerate(sorted_results[:3]):
            chain_str = " → ".join(result["chain"])
            gain = result["metrics"]["information_gain"]
            print(f"{i+1}. {chain_str}")
            print(f"   Problem: {result['problem'][:50]}...")
            print(f"   Information gain: {gain}")
            
        # Analyze patterns
        print("\n🔍 Patterns Discovered:")
        
        # Which models work best together?
        pair_scores = {}
        for result in results:
            chain = result["chain"]
            gain = result["metrics"]["information_gain"]
            for i in range(len(chain)-1):
                pair = (chain[i], chain[i+1])
                if pair not in pair_scores:
                    pair_scores[pair] = []
                pair_scores[pair].append(gain)
                
        # Average scores for each pair
        avg_pair_scores = {}
        for pair, scores in pair_scores.items():
            avg_pair_scores[pair] = sum(scores) / len(scores)
            
        # Top collaborating pairs
        top_pairs = sorted(avg_pair_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print("\nBest Collaborating Pairs:")
        for pair, score in top_pairs:
            print(f"  {pair[0]} → {pair[1]}: avg gain {score:.0f}")
            
        # Save report
        report = {
            "generated": datetime.now().isoformat(),
            "total_chains": len(results),
            "top_chains": [
                {
                    "chain": r["chain"],
                    "problem": r["problem"],
                    "information_gain": r["metrics"]["information_gain"]
                }
                for r in sorted_results[:5]
            ],
            "best_pairs": [
                {"pair": list(p), "avg_gain": s}
                for p, s in top_pairs
            ]
        }
        
        with open(f"{self.chains_dir}value_creation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        print("\nValue creation patterns reveal how AI models complement each other!")
        print("Report saved to value_chains/value_creation_report.json")


if __name__ == "__main__":
    print("Starting Value Creation Chain experiments...")
    print("Models will collaborate to solve real problems\n")
    
    chain = ValueCreationChain()
    chain.run_experiments()