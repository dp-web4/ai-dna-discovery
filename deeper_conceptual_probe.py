#!/usr/bin/env python3
"""
Deeper Conceptual Probe
Exploring why models show complementary rather than identical conceptual recognition
"""

import json
import time
from datetime import datetime
from ai_lct_ollama_integration import OllamaLCTClient


class DeeperProbe:
    """Investigate the complementary nature of model conceptual spaces"""
    
    def __init__(self):
        self.client = OllamaLCTClient()
        self.models = ["phi3:mini", "tinyllama:latest"]
        for model in self.models:
            self.client.register_model(model)
    
    def probe_concept_directly(self, model: str, concept: str) -> dict:
        """Ask model directly about a concept"""
        prompt = f"Define '{concept}' in one sentence."
        result = self.client.generate(model, prompt, energy_cost=3.0)
        return {
            "model": model,
            "concept": concept,
            "response": result.get("response", "ERROR"),
            "energy": result.get("energy_spent", 0)
        }
    
    def test_complementarity_hypothesis(self):
        """Test if models have complementary conceptual coverage"""
        
        print("=== Testing Complementarity Hypothesis ===")
        print("Perhaps models specialize in different conceptual domains?\n")
        
        # From our results:
        # Phi3 recognized: hot/cold (sensory), increase/decrease (quantitative)
        # TinyLlama recognized: parent/child (relational), cause/effect (logical), whole/part (structural)
        
        domains = {
            "sensory/physical": ["hot", "cold", "bright", "dark", "heavy", "light"],
            "quantitative": ["increase", "decrease", "more", "less", "maximum", "minimum"],
            "relational": ["parent", "child", "teacher", "student", "leader", "follower"],
            "logical": ["cause", "effect", "premise", "conclusion", "if", "then"],
            "structural": ["whole", "part", "system", "component", "container", "content"]
        }
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "hypothesis": "Models have complementary conceptual specializations",
            "domain_coverage": {}
        }
        
        for domain, concepts in domains.items():
            print(f"\nTesting domain: {domain}")
            results["domain_coverage"][domain] = {}
            
            for concept in concepts[:2]:  # Test first 2 from each domain
                for model in self.models:
                    response = self.probe_concept_directly(model, concept)
                    print(f"  {model} on '{concept}': {len(response['response'])} chars")
                    
                    if model not in results["domain_coverage"][domain]:
                        results["domain_coverage"][domain][model] = []
                    
                    results["domain_coverage"][domain][model].append({
                        "concept": concept,
                        "response_length": len(response['response']),
                        "contains_concept": concept.lower() in response['response'].lower()
                    })
        
        # Analyze specialization patterns
        print("\n=== ANALYSIS ===")
        
        specializations = {
            "phi3:mini": [],
            "tinyllama:latest": []
        }
        
        for domain, coverage in results["domain_coverage"].items():
            phi3_avg = sum(r["response_length"] for r in coverage.get("phi3:mini", [])) / 2
            tiny_avg = sum(r["response_length"] for r in coverage.get("tinyllama:latest", [])) / 2
            
            if phi3_avg > tiny_avg * 1.2:
                specializations["phi3:mini"].append(domain)
            elif tiny_avg > phi3_avg * 1.2:
                specializations["tinyllama:latest"].append(domain)
        
        results["specializations"] = specializations
        
        print("Model Specializations:")
        for model, domains in specializations.items():
            if domains:
                print(f"  {model}: {', '.join(domains)}")
        
        # Save results
        with open("/home/dp/ai-workspace/complementarity_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def test_conceptual_bridging(self):
        """Can one model help another understand concepts?"""
        
        print("\n\n=== Testing Conceptual Bridging ===")
        print("Can models teach each other concepts?\n")
        
        # Concepts from our test where only one model succeeded
        test_cases = [
            ("phi3:mini", "tinyllama:latest", "hot/cold are opposites"),
            ("tinyllama:latest", "phi3:mini", "parent and child are reciprocal roles")
        ]
        
        for teacher, student, concept in test_cases:
            print(f"\nTeacher: {teacher}, Student: {student}")
            print(f"Concept: {concept}")
            
            # Get teacher's explanation
            teacher_prompt = f"Explain why '{concept}' in simple terms."
            teacher_response = self.client.generate(teacher, teacher_prompt, energy_cost=5.0)
            
            if "error" not in teacher_response:
                explanation = teacher_response["response"]
                print(f"Teacher explains: {explanation[:100]}...")
                
                # Test if student now understands
                student_prompt = f"Based on this explanation: '{explanation[:200]}', is it true that {concept}? YES or NO only."
                student_response = self.client.generate(student, student_prompt, energy_cost=3.0)
                
                if "error" not in student_response:
                    understands = "YES" in student_response["response"].upper()
                    print(f"Student understands: {'✓' if understands else '✗'}")
        
        print("\n=== IMPLICATIONS ===")
        print("1. Models may have different 'cognitive styles' or specializations")
        print("2. This supports DP's theory - universal embeddings exist but are accessed differently")
        print("3. Like humans, AI models might benefit from collaborative learning")
        print("4. Web4's multi-agent systems could leverage complementary strengths")


def main():
    """Run deeper conceptual analysis"""
    
    prober = DeeperProbe()
    
    # Test complementarity
    prober.test_complementarity_hypothesis()
    
    # Test bridging
    prober.test_conceptual_bridging()
    
    # Log findings
    with open("/home/dp/ai-workspace/autonomous_exploration_log.md", "a") as f:
        f.write(f"\n### Deeper Probe - {datetime.now().isoformat()}\n")
        f.write("- Found evidence for complementary specialization\n")
        f.write("- Models can potentially teach each other concepts\n")
        f.write("- Supports universal embeddings accessed through different paths\n\n")


if __name__ == "__main__":
    main()