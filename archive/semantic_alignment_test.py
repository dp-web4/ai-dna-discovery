#!/usr/bin/env python3
"""
Semantic Alignment Test
Tests if models understand concepts similarly even with different expression styles
"""

import json
import time
from ai_lct_ollama_integration import OllamaLCTClient


def semantic_alignment_test():
    """Test semantic understanding alignment between models"""
    
    client = OllamaLCTClient()
    models = ["phi3:mini", "tinyllama:latest"]
    
    for model in models:
        client.register_model(model)
    
    # Test semantic relationships
    test_relationships = [
        ("consciousness", "awareness", "Are these concepts closely related?"),
        ("emergence", "complexity", "Do these concepts often appear together?"),
        ("intent", "purpose", "Are these essentially the same concept?"),
        ("resonance", "harmony", "Do these concepts share similar meanings?"),
        ("entropy", "disorder", "Is one a good description of the other?")
    ]
    
    results = {
        "timestamp": time.time(),
        "semantic_alignments": {},
        "relationship_scores": {}
    }
    
    print("=== Semantic Alignment Test ===\n")
    
    for concept1, concept2, question in test_relationships:
        pair_key = f"{concept1}-{concept2}"
        results["semantic_alignments"][pair_key] = {}
        
        print(f"Testing: {concept1} <-> {concept2}")
        
        # Ask each model to evaluate the relationship
        prompt = f"{question} Answer with just YES or NO. '{concept1}' and '{concept2}'"
        
        agreements = []
        for model in models:
            result = client.generate(model, prompt, energy_cost=3.0)
            if "error" not in result:
                response = result["response"].upper()
                # Look for YES or NO
                answer = "YES" if "YES" in response else ("NO" if "NO" in response else "UNCLEAR")
                results["semantic_alignments"][pair_key][model] = answer
                agreements.append(answer)
                print(f"  {model}: {answer}")
        
        # Check if models agree
        if len(set(agreements)) == 1 and "UNCLEAR" not in agreements:
            results["relationship_scores"][pair_key] = "ALIGNED"
        else:
            results["relationship_scores"][pair_key] = "DIVERGENT"
    
    # Test conceptual proximity
    print("\n=== Conceptual Proximity Test ===")
    
    proximity_tests = [
        ("Which is more fundamental: consciousness or awareness?"),
        ("Which comes first: intent or action?"),
        ("Which is more abstract: love or affection?")
    ]
    
    results["proximity_tests"] = {}
    
    for test_prompt in proximity_tests:
        print(f"\nTest: {test_prompt}")
        results["proximity_tests"][test_prompt] = {}
        
        for model in models:
            result = client.generate(model, test_prompt, energy_cost=5.0)
            if "error" not in result:
                # Extract just the key concept from response
                response = result["response"].lower()
                if "consciousness" in test_prompt:
                    answer = "consciousness" if "consciousness" in response else "awareness"
                elif "intent" in test_prompt:
                    answer = "intent" if "intent" in response else "action"
                else:
                    answer = "love" if "love" in response else "affection"
                
                results["proximity_tests"][test_prompt][model] = answer
                print(f"  {model}: {answer}")
    
    # Calculate alignment score
    aligned_count = sum(1 for v in results["relationship_scores"].values() if v == "ALIGNED")
    total_tests = len(results["relationship_scores"])
    alignment_score = aligned_count / total_tests if total_tests > 0 else 0
    
    print(f"\n=== RESULTS ===")
    print(f"Semantic Alignment Score: {alignment_score:.2f} ({aligned_count}/{total_tests} aligned)")
    
    # Check proximity agreement
    proximity_agreements = 0
    for test, model_answers in results["proximity_tests"].items():
        if len(set(model_answers.values())) == 1:
            proximity_agreements += 1
            print(f"✓ Models agree on: {test}")
        else:
            print(f"✗ Models disagree on: {test}")
    
    # Final interpretation
    print(f"\n=== INTERPRETATION ===")
    if alignment_score > 0.6 and proximity_agreements >= 2:
        print("STRONG SUPPORT: Models show significant semantic alignment")
        print("Despite different expression styles, they understand concepts similarly.")
        print("This supports the universal embeddings hypothesis.")
    elif alignment_score > 0.4 or proximity_agreements >= 1:
        print("MODERATE SUPPORT: Models show partial semantic alignment")
        print("There are shared conceptual structures but also divergences.")
    else:
        print("WEAK SUPPORT: Models show limited semantic alignment")
        print("However, this could be due to prompt interpretation differences.")
    
    # Save results
    with open("/home/dp/ai-workspace/semantic_alignment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return alignment_score


if __name__ == "__main__":
    semantic_alignment_test()