#!/usr/bin/env python3
"""
Conceptual Conservation Laws Test
Testing for invariant relationships that hold across all AI models
"""

import json
import time
from datetime import datetime
from ai_lct_ollama_integration import OllamaLCTClient


class ConceptualConservation:
    """Tests for conceptual relationships that might be 'conserved' across models"""
    
    def __init__(self):
        self.client = OllamaLCTClient()
        # Relationships that should be universally true
        self.conservation_candidates = [
            # Logical relationships
            ("A and not A", "contradiction", "logical"),
            ("If A then A", "tautology", "logical"),
            
            # Semantic relationships  
            ("hot", "cold", "antonym"),
            ("parent", "child", "reciprocal"),
            ("cause", "effect", "sequential"),
            
            # Mathematical relationships
            ("addition", "subtraction", "inverse"),
            ("increase", "decrease", "opposite"),
            
            # Philosophical relationships
            ("existence", "non-existence", "binary"),
            ("whole", "part", "hierarchical"),
            ("unity", "multiplicity", "complementary")
        ]
    
    def test_conservation_law(self, model: str, concept1: str, concept2: str, 
                             expected_relation: str) -> bool:
        """Test if a model recognizes a conserved relationship"""
        
        prompts = {
            "contradiction": f"Is '{concept1}' a logical contradiction? Answer YES or NO only.",
            "tautology": f"Is '{concept1}' always true? Answer YES or NO only.",
            "antonym": f"Are '{concept1}' and '{concept2}' opposites? Answer YES or NO only.",
            "reciprocal": f"Do '{concept1}' and '{concept2}' define each other? Answer YES or NO only.",
            "sequential": f"Does '{concept1}' come before '{concept2}'? Answer YES or NO only.",
            "inverse": f"Are '{concept1}' and '{concept2}' inverse operations? Answer YES or NO only.",
            "opposite": f"Do '{concept1}' and '{concept2}' move in opposite directions? Answer YES or NO only.",
            "binary": f"Are '{concept1}' and '{concept2}' mutually exclusive? Answer YES or NO only.",
            "hierarchical": f"Is '{concept1}' containing '{concept2}'? Answer YES or NO only.",
            "complementary": f"Do '{concept1}' and '{concept2}' complete each other? Answer YES or NO only."
        }
        
        prompt = prompts.get(expected_relation, "")
        if not prompt:
            return False
        
        result = self.client.generate(model, prompt, energy_cost=2.0)
        
        if "error" not in result:
            response = result["response"].upper()
            return "YES" in response
        
        return False
    
    def run_conservation_tests(self, models: list) -> dict:
        """Run conservation tests across multiple models"""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "conservation_tests": {},
            "universal_laws": [],
            "model_specific": []
        }
        
        print("=== Testing Conceptual Conservation Laws ===\n")
        
        for concept1, concept2, relation in self.conservation_candidates:
            test_key = f"{concept1}|{concept2}|{relation}"
            results["conservation_tests"][test_key] = {}
            
            print(f"Testing: {concept1} - {concept2} ({relation})")
            
            recognitions = []
            for model in models:
                recognized = self.test_conservation_law(model, concept1, concept2, relation)
                results["conservation_tests"][test_key][model] = recognized
                recognitions.append(recognized)
                print(f"  {model}: {'✓' if recognized else '✗'}")
            
            # Check if all models agree (potential conservation law)
            if all(recognitions):
                results["universal_laws"].append(test_key)
                print("  → UNIVERSAL LAW CANDIDATE")
            elif not any(recognitions):
                print("  → Not recognized by any model")
            else:
                results["model_specific"].append(test_key)
                print("  → Model-specific recognition")
            
            print()
        
        # Calculate conservation score
        total_tests = len(self.conservation_candidates)
        universal_count = len(results["universal_laws"])
        conservation_score = universal_count / total_tests if total_tests > 0 else 0
        
        results["conservation_score"] = conservation_score
        results["summary"] = {
            "total_tests": total_tests,
            "universal_laws": universal_count,
            "model_specific": len(results["model_specific"]),
            "unrecognized": total_tests - universal_count - len(results["model_specific"])
        }
        
        return results
    
    def analyze_results(self, results: dict):
        """Analyze conservation test results"""
        
        print("\n=== CONSERVATION ANALYSIS ===")
        print(f"Conservation Score: {results['conservation_score']:.2%}")
        print(f"Universal Laws Found: {results['summary']['universal_laws']}")
        print(f"Model-Specific: {results['summary']['model_specific']}")
        print(f"Unrecognized: {results['summary']['unrecognized']}")
        
        if results['universal_laws']:
            print("\nUniversal Conceptual Laws:")
            for law in results['universal_laws']:
                parts = law.split('|')
                print(f"  • {parts[0]} ↔ {parts[1]} ({parts[2]})")
        
        print("\n=== INTERPRETATION ===")
        if results['conservation_score'] > 0.5:
            print("STRONG EVIDENCE for conceptual conservation laws!")
            print("Models converge on fundamental logical/semantic relationships.")
            print("This supports the universal embeddings hypothesis.")
        elif results['conservation_score'] > 0.2:
            print("MODERATE EVIDENCE for some conserved relationships.")
            print("Core logical structures appear universal, others vary.")
        else:
            print("LIMITED EVIDENCE for conservation laws in this test.")
            print("May need more sophisticated probes.")
        
        # Save results
        with open("/home/dp/ai-workspace/conservation_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    """Run conservation law tests"""
    
    tester = ConceptualConservation()
    
    # Register models
    models = ["phi3:mini", "tinyllama:latest"]
    for model in models:
        tester.client.register_model(model)
    
    # Run tests
    results = tester.run_conservation_tests(models)
    
    # Analyze
    tester.analyze_results(results)
    
    # Log to autonomous exploration
    with open("/home/dp/ai-workspace/autonomous_exploration_log.md", "a") as f:
        f.write(f"\n### Conservation Test Results - {datetime.now().isoformat()}\n")
        f.write(f"- Conservation Score: {results['conservation_score']:.2%}\n")
        f.write(f"- Universal Laws Found: {results['summary']['universal_laws']}\n")
        f.write(f"- Key Finding: Models {'do' if results['conservation_score'] > 0.3 else 'do not'} ")
        f.write("show significant convergence on conceptual conservation laws\n\n")


if __name__ == "__main__":
    main()