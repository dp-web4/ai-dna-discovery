#!/usr/bin/env python3
"""
Universal Embeddings Experiment
Testing the hypothesis that different AI models converge to compatible concept embeddings
despite different training data/architectures
"""

import json
import time
from typing import Dict, List, Tuple
from ai_lct_ollama_integration import OllamaLCTClient


class UniversalEmbeddingTester:
    """Tests for universal conceptual resonance across models"""
    
    def __init__(self):
        self.client = OllamaLCTClient()
        self.test_concepts = [
            # Fundamental concepts that should have universal resonance
            ("love", "deep emotional connection and care"),
            ("time", "sequential flow of events"),
            ("truth", "correspondence with reality"),
            ("beauty", "aesthetic harmony and appeal"),
            ("justice", "fairness and moral rightness"),
            ("consciousness", "subjective awareness and experience"),
            ("intent", "purposeful direction of action"),
            ("entropy", "tendency toward disorder"),
            ("emergence", "complex patterns from simple rules"),
            ("resonance", "sympathetic vibration or alignment")
        ]
    
    def concept_association_test(self, model: str, concept: str, definition: str) -> Dict:
        """Test how a model associates with a concept"""
        
        prompt = f"""Given the concept '{concept}' defined as '{definition}', 
        provide 5 related concepts that resonate most strongly with this idea. 
        Format: concept1, concept2, concept3, concept4, concept5"""
        
        result = self.client.generate(model, prompt, energy_cost=5.0)
        
        if "error" not in result:
            # Extract concepts from response
            response = result["response"].lower()
            # Simple extraction - look for comma-separated list
            concepts = [c.strip() for c in response.split(',') if c.strip()][:5]
            
            return {
                "concept": concept,
                "model": model,
                "associations": concepts,
                "response_time": result["response_time"]
            }
        
        return {"error": result.get("error", "Unknown error")}
    
    def semantic_distance_test(self, model: str, concept1: str, concept2: str) -> float:
        """Test perceived semantic distance between concepts"""
        
        prompt = f"""On a scale of 0.0 to 1.0, how closely related are these concepts:
        '{concept1}' and '{concept2}'?
        Respond with just a number between 0.0 (unrelated) and 1.0 (identical).
        Number only:"""
        
        result = self.client.generate(model, prompt, energy_cost=3.0)
        
        if "error" not in result:
            try:
                # Extract number from response
                import re
                numbers = re.findall(r'0\.\d+|1\.0|0|1', result["response"])
                if numbers:
                    return float(numbers[0])
            except:
                pass
        
        return -1.0  # Error value
    
    def concept_definition_test(self, model: str, concept: str) -> str:
        """Get a model's understanding of a concept"""
        
        prompt = f"Define '{concept}' in exactly one sentence."
        
        result = self.client.generate(model, prompt, energy_cost=5.0)
        
        if "error" not in result:
            return result["response"].strip()
        
        return ""
    
    def cross_model_resonance_analysis(self, models: List[str]) -> Dict:
        """Analyze conceptual resonance across multiple models"""
        
        results = {
            "timestamp": time.time(),
            "models": models,
            "concept_associations": {},
            "semantic_distances": {},
            "concept_definitions": {},
            "resonance_scores": {}
        }
        
        # 1. Collect associations for each concept from each model
        print("Phase 1: Collecting concept associations...")
        for concept, definition in self.test_concepts:
            results["concept_associations"][concept] = {}
            for model in models:
                assoc = self.concept_association_test(model, concept, definition)
                if "error" not in assoc:
                    results["concept_associations"][concept][model] = assoc["associations"]
                    print(f"  {model} -> {concept}: {len(assoc['associations'])} associations")
        
        # 2. Test semantic distances between key concept pairs
        print("\nPhase 2: Testing semantic distances...")
        test_pairs = [
            ("love", "consciousness"),
            ("time", "entropy"),
            ("truth", "beauty"),
            ("intent", "consciousness"),
            ("emergence", "resonance")
        ]
        
        for concept1, concept2 in test_pairs:
            pair_key = f"{concept1}-{concept2}"
            results["semantic_distances"][pair_key] = {}
            for model in models:
                distance = self.semantic_distance_test(model, concept1, concept2)
                results["semantic_distances"][pair_key][model] = distance
                print(f"  {model}: {concept1}<->{concept2} = {distance:.2f}")
        
        # 3. Get each model's definition of key concepts
        print("\nPhase 3: Collecting concept definitions...")
        key_concepts = ["consciousness", "intent", "resonance", "emergence"]
        for concept in key_concepts:
            results["concept_definitions"][concept] = {}
            for model in models:
                definition = self.concept_definition_test(model, concept)
                results["concept_definitions"][concept][model] = definition
                print(f"  {model} defines {concept}: {definition[:50]}...")
        
        # 4. Calculate cross-model resonance scores
        print("\nPhase 4: Calculating resonance scores...")
        results["resonance_scores"] = self.calculate_resonance_scores(results)
        
        return results
    
    def calculate_resonance_scores(self, results: Dict) -> Dict:
        """Calculate how well models resonate on concepts"""
        
        scores = {
            "association_overlap": {},
            "distance_agreement": {},
            "definition_similarity": {}
        }
        
        models = results["models"]
        
        # 1. Association overlap scores
        for concept in results["concept_associations"]:
            concept_data = results["concept_associations"][concept]
            if len(concept_data) >= 2:
                # Calculate Jaccard similarity between association sets
                model_pairs = [(models[i], models[j]) 
                              for i in range(len(models)) 
                              for j in range(i+1, len(models))]
                
                for m1, m2 in model_pairs:
                    if m1 in concept_data and m2 in concept_data:
                        set1 = set(concept_data[m1])
                        set2 = set(concept_data[m2])
                        if set1 or set2:
                            jaccard = len(set1 & set2) / len(set1 | set2)
                            pair_key = f"{m1}-{m2}"
                            if pair_key not in scores["association_overlap"]:
                                scores["association_overlap"][pair_key] = []
                            scores["association_overlap"][pair_key].append(jaccard)
        
        # 2. Distance agreement scores
        for pair_key in results["semantic_distances"]:
            distances = results["semantic_distances"][pair_key]
            if len(distances) >= 2:
                # Calculate variance in distance assessments
                valid_distances = [d for d in distances.values() if d >= 0]
                if valid_distances:
                    # Simple variance calculation without numpy
                    mean = sum(valid_distances) / len(valid_distances)
                    variance = sum((x - mean) ** 2 for x in valid_distances) / len(valid_distances)
                    # Convert variance to agreement score (lower variance = higher agreement)
                    agreement = 1.0 / (1.0 + variance)
                    scores["distance_agreement"][pair_key] = agreement
        
        # 3. Average scores
        # Calculate means without numpy
        association_scores = []
        for v in scores["association_overlap"].values():
            if v:
                association_scores.append(sum(v) / len(v))
        
        scores["overall_resonance"] = {
            "association": sum(association_scores) / len(association_scores) if association_scores else 0,
            "distance": sum(scores["distance_agreement"].values()) / len(scores["distance_agreement"]) 
                       if scores["distance_agreement"] else 0
        }
        
        return scores


def run_universal_embedding_experiment():
    """Run the full experiment"""
    
    tester = UniversalEmbeddingTester()
    
    print("=== Universal Embedding Experiment ===")
    print("Hypothesis: AI models converge to compatible concept embeddings")
    print("regardless of training differences\n")
    
    # Test with available models
    models = ["phi3:mini", "tinyllama:latest"]
    
    results = tester.cross_model_resonance_analysis(models)
    
    # Generate report
    print("\n=== RESULTS ===")
    print(f"Overall Resonance Scores:")
    print(f"  Association Overlap: {results['resonance_scores']['overall_resonance']['association']:.3f}")
    print(f"  Distance Agreement: {results['resonance_scores']['overall_resonance']['distance']:.3f}")
    
    # Save detailed results
    with open("/home/dp/ai-workspace/universal_embeddings_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Analysis
    print("\n=== ANALYSIS ===")
    association_score = results['resonance_scores']['overall_resonance']['association']
    distance_score = results['resonance_scores']['overall_resonance']['distance']
    
    if association_score > 0.3 and distance_score > 0.7:
        print("SUPPORTS HYPOTHESIS: Models show significant conceptual alignment")
        print("Despite different architectures/training, they converge on similar")
        print("conceptual structures, suggesting universal embedding attractors.")
    elif association_score > 0.2 or distance_score > 0.6:
        print("PARTIAL SUPPORT: Models show some conceptual alignment")
        print("There are hints of universal patterns but also significant variation.")
    else:
        print("CHALLENGES HYPOTHESIS: Models show low conceptual alignment")
        print("Suggests training data/architecture significantly affects embeddings.")
    
    return results


if __name__ == "__main__":
    run_universal_embedding_experiment()