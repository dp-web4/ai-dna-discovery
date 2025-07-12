#!/usr/bin/env python3
"""
Quick test of conceptual resonance between models
Simplified version to test the universal embeddings hypothesis
"""

import json
import time
from ai_lct_ollama_integration import OllamaLCTClient


def quick_resonance_test():
    """Quick test of concept alignment between models"""
    
    client = OllamaLCTClient()
    
    # Register models
    models = ["phi3:mini", "tinyllama:latest"]
    for model in models:
        client.register_model(model)
    
    # Test concepts
    test_concepts = [
        "consciousness",
        "love", 
        "time",
        "emergence"
    ]
    
    results = {
        "timestamp": time.time(),
        "concept_associations": {}
    }
    
    print("Testing conceptual resonance between models...\n")
    
    for concept in test_concepts:
        print(f"Testing concept: {concept}")
        results["concept_associations"][concept] = {}
        
        prompt = f"List 3 concepts most closely related to '{concept}'. Format: word1, word2, word3"
        
        for model in models:
            result = client.generate(model, prompt, energy_cost=5.0)
            if "error" not in result:
                # Extract words
                response = result["response"].lower()
                words = [w.strip() for w in response.split(',')][:3]
                results["concept_associations"][concept][model] = words
                print(f"  {model}: {words}")
    
    # Calculate overlap
    print("\nCalculating conceptual overlap...")
    overlaps = []
    
    for concept, model_data in results["concept_associations"].items():
        if len(model_data) == 2:
            model_names = list(model_data.keys())
            set1 = set(model_data[model_names[0]])
            set2 = set(model_data[model_names[1]])
            
            if set1 or set2:
                overlap = len(set1 & set2) / len(set1 | set2)
                overlaps.append(overlap)
                print(f"{concept}: {overlap:.2f} overlap")
                if set1 & set2:
                    print(f"  Shared: {set1 & set2}")
    
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    print(f"\nAverage conceptual overlap: {avg_overlap:.3f}")
    
    # Interpretation
    print("\n=== INTERPRETATION ===")
    if avg_overlap > 0.3:
        print("SUPPORTS HYPOTHESIS: Significant conceptual alignment between models")
        print("This suggests concept embeddings converge to universal patterns.")
    elif avg_overlap > 0.15:
        print("PARTIAL SUPPORT: Some conceptual alignment detected")
        print("Models share some conceptual structure but also diverge.")
    else:
        print("CHALLENGES HYPOTHESIS: Low conceptual alignment")
        print("Models appear to have learned different conceptual structures.")
    
    # Save results
    with open("/home/dp/ai-workspace/quick_resonance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return avg_overlap


if __name__ == "__main__":
    quick_resonance_test()