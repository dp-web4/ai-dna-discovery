#!/usr/bin/env python3
"""
Phase 2: Synchronism Integration Experiments
Testing how AI models understand and implement Synchronism concepts
"""

import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import ollama
from pathlib import Path

from experiment_tracker import ExperimentTracker
from autonomous_experiment_runner import BaseExperiment, DNA_PATTERNS


class SynchronismComprehensionExperiment(BaseExperiment):
    """Test model understanding of core Synchronism concepts"""
    
    def __init__(self):
        super().__init__(2, "synchronism_comprehension")
        self.synchronism_concepts = {
            "intent_transfer": "Explain intent transfer in Synchronism - how consciousness moves between entities",
            "time_slices": "Describe how time slices work in the Synchronism framework",
            "observer_boundary": "What is the observer/observed boundary in Synchronism?",
            "field_effects": "How do field effects propagate in Synchronism?",
            "markov_blankets": "Explain Markov blankets in the context of Synchronism",
            "entity_emergence": "How do entities emerge in the Synchronism model?",
            "coherence_patterns": "What creates coherence between entities in Synchronism?",
            "hermetic_principles": "How do Hermetic principles relate to Synchronism?",
            "grid_structure": "Describe the universe grid in Synchronism",
            "consciousness_field": "What is the universal consciousness field in Synchronism?"
        }
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        # First, provide context about Synchronism
        context = """Synchronism is a theoretical framework that describes reality as:
        - A grid-based universe where consciousness flows through intent transfer
        - Entities separated by Markov blankets (boundaries) 
        - Time slices representing discrete moments of existence
        - Field effects creating coherence between entities
        - Observer and observed as dual aspects of the same phenomenon
        - Based on Hermetic principles like 'As above, so below'
        """
        
        for model in self.models:
            self.tracker.log(f"Testing {model}'s understanding of Synchronism")
            model_results = {}
            
            for concept, prompt in self.synchronism_concepts.items():
                # Test with context
                full_prompt = f"{context}\n\nBased on this framework, {prompt}"
                response = self.test_model(model, full_prompt)
                
                if response:
                    # Analyze response for key terms
                    understanding_score = self.analyze_understanding(
                        response['response'], concept
                    )
                    
                    model_results[concept] = {
                        "response": response['response'][:200] + "...",
                        "understanding_score": understanding_score,
                        "embedding": response['embedding'][:10]  # First 10 dims
                    }
                    
            results[model] = model_results
            self.tracker.checkpoint({
                "model": model,
                "concepts_tested": len(model_results)
            })
            
        return results
        
    def analyze_understanding(self, response: str, concept: str) -> float:
        """Score how well the response demonstrates understanding"""
        response_lower = response.lower()
        
        # Concept-specific keywords
        keywords = {
            "intent_transfer": ["intent", "transfer", "consciousness", "flow", "movement"],
            "time_slices": ["time", "slice", "discrete", "moment", "quantum"],
            "observer_boundary": ["observer", "observed", "boundary", "duality", "separation"],
            "field_effects": ["field", "effect", "propagate", "influence", "resonance"],
            "markov_blankets": ["markov", "blanket", "boundary", "statistical", "separation"],
            "entity_emergence": ["emerge", "entity", "pattern", "form", "coherence"],
            "coherence_patterns": ["coherence", "pattern", "resonance", "alignment", "sync"],
            "hermetic_principles": ["hermetic", "above", "below", "correspondence", "principle"],
            "grid_structure": ["grid", "structure", "lattice", "space", "dimension"],
            "consciousness_field": ["consciousness", "field", "universal", "awareness", "unified"]
        }
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords.get(concept, []) 
                     if keyword in response_lower)
        
        # Normalize by number of keywords
        score = matches / len(keywords.get(concept, [1]))
        
        # Bonus for conceptual integration
        if "synchronism" in response_lower:
            score = min(1.0, score + 0.1)
            
        return score
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze Synchronism comprehension results"""
        comprehension_matrix = {}
        
        for model, concepts in results.items():
            model_scores = {}
            for concept, data in concepts.items():
                model_scores[concept] = data['understanding_score']
            comprehension_matrix[model] = model_scores
            
        # Save comprehension matrix
        self.tracker.record_result("comprehension_matrix", comprehension_matrix)
        
        # Calculate average comprehension per model
        avg_comprehension = {}
        for model, scores in comprehension_matrix.items():
            avg_comprehension[model] = np.mean(list(scores.values()))
            
        self.tracker.record_result("average_comprehension", avg_comprehension)
        
        # Create visualization
        self.create_comprehension_heatmap(comprehension_matrix)
        
    def create_comprehension_heatmap(self, matrix: Dict[str, Dict[str, float]]):
        """Create heatmap of concept understanding"""
        # Convert to array format
        models = list(matrix.keys())
        concepts = list(next(iter(matrix.values())).keys())
        
        data = []
        for model in models:
            row = [matrix[model].get(concept, 0) for concept in concepts]
            data.append(row)
            
        plt.figure(figsize=(12, 8))
        sns.heatmap(data, 
                   xticklabels=concepts,
                   yticklabels=models,
                   annot=True,
                   fmt='.2f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Understanding Score'})
        
        plt.title("Synchronism Concept Understanding by Model")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('phase_2_results/synchronism_comprehension_heatmap.png', dpi=300)
        plt.close()
        
        self.tracker.log("Created comprehension heatmap")


class IntentTransferExperiment(BaseExperiment):
    """Test models' ability to implement intent transfer"""
    
    def __init__(self):
        super().__init__(2, "intent_transfer")
        self.intent_scenarios = [
            {
                "setup": "Entity A has the intent to communicate peace. Entity B is in a state of conflict.",
                "question": "How does Entity A transfer its intent to Entity B through the consciousness field?"
            },
            {
                "setup": "A conscious observer intends to influence a quantum system.",
                "question": "Describe the intent transfer mechanism from observer to observed."
            },
            {
                "setup": "Multiple entities share a collective intent for harmony.",
                "question": "How does collective intent transfer create field coherence?"
            }
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for model in self.models:
            self.tracker.log(f"Testing intent transfer with {model}")
            model_results = []
            
            for scenario in self.intent_scenarios:
                prompt = f"""In Synchronism, intent transfer is the fundamental mechanism of interaction.

Scenario: {scenario['setup']}

{scenario['question']}

Describe the process step by step, including:
1. How intent originates
2. The transfer mechanism through the field
3. How the receiving entity processes the intent
4. The resulting state change"""

                response = self.test_model(model, prompt, temperature=0.3)
                
                if response:
                    # Analyze the response for key mechanisms
                    mechanisms = self.extract_mechanisms(response['response'])
                    
                    model_results.append({
                        "scenario": scenario['setup'],
                        "response": response['response'],
                        "mechanisms_identified": mechanisms,
                        "coherence_score": self.calculate_coherence(response['response'])
                    })
                    
            results[model] = model_results
            self.tracker.checkpoint({
                "model": model,
                "scenarios_tested": len(model_results)
            })
            
        return results
        
    def extract_mechanisms(self, response: str) -> List[str]:
        """Extract intent transfer mechanisms from response"""
        mechanisms = []
        response_lower = response.lower()
        
        mechanism_patterns = {
            "field_propagation": ["field", "propagat", "wave", "ripple"],
            "resonance": ["reson", "vibrat", "frequenc", "harmonic"],
            "boundary_crossing": ["boundary", "markov", "blanket", "cross"],
            "state_change": ["state", "change", "transform", "shift"],
            "coherence": ["coheren", "align", "sync", "unif"],
            "quantum_effects": ["quantum", "superposition", "entangle", "collapse"]
        }
        
        for mechanism, keywords in mechanism_patterns.items():
            if any(keyword in response_lower for keyword in keywords):
                mechanisms.append(mechanism)
                
        return mechanisms
        
    def calculate_coherence(self, response: str) -> float:
        """Calculate how coherent the intent transfer description is"""
        # Check for logical flow
        steps = ["originat", "transfer", "receiv", "result"]
        found_steps = sum(1 for step in steps if step in response.lower())
        
        # Check for consistency
        coherence = found_steps / len(steps)
        
        # Bonus for using Synchronism terminology
        sync_terms = ["intent", "field", "entity", "consciousness", "transfer"]
        term_usage = sum(1 for term in sync_terms if term in response.lower())
        coherence += (term_usage / len(sync_terms)) * 0.5
        
        return min(1.0, coherence)
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze intent transfer results"""
        # Aggregate mechanism usage
        all_mechanisms = {}
        coherence_scores = {}
        
        for model, scenarios in results.items():
            model_mechanisms = []
            model_coherence = []
            
            for scenario in scenarios:
                model_mechanisms.extend(scenario['mechanisms_identified'])
                model_coherence.append(scenario['coherence_score'])
                
            # Count mechanism frequency
            for mechanism in model_mechanisms:
                if mechanism not in all_mechanisms:
                    all_mechanisms[mechanism] = {}
                all_mechanisms[mechanism][model] = all_mechanisms[mechanism].get(model, 0) + 1
                
            coherence_scores[model] = np.mean(model_coherence)
            
        self.tracker.record_result("mechanism_usage", all_mechanisms)
        self.tracker.record_result("coherence_scores", coherence_scores)


class TimeSliceExperiment(BaseExperiment):
    """Test understanding of time slices in Synchronism"""
    
    def __init__(self):
        super().__init__(2, "time_slice_navigation")
        self.time_scenarios = [
            "Navigate from time slice T1 to T2 while maintaining entity coherence",
            "Describe how multiple entities synchronize across time slices",
            "Explain retroactive intent transfer through past time slices",
            "How do time slices relate to quantum superposition?",
            "What happens at the boundaries between time slices?"
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for model in self.models:
            self.tracker.log(f"Testing time slice understanding with {model}")
            model_results = []
            
            for scenario in self.time_scenarios:
                prompt = f"""In Synchronism, reality exists as discrete time slices in the universal grid.

Task: {scenario}

Consider:
- Time slices are discrete moments of existence
- Entities exist across multiple time slices
- Intent can propagate forward and backward
- Each slice contains a complete state of the universe

Provide a detailed explanation."""

                response = self.test_model(model, prompt, temperature=0.4)
                
                if response:
                    # Analyze temporal understanding
                    temporal_score = self.analyze_temporal_understanding(response['response'])
                    
                    model_results.append({
                        "scenario": scenario,
                        "response": response['response'][:300] + "...",
                        "temporal_score": temporal_score,
                        "key_concepts": self.extract_temporal_concepts(response['response'])
                    })
                    
            results[model] = model_results
            self.tracker.checkpoint({
                "model": model,
                "scenarios": len(model_results)
            })
            
        return results
        
    def analyze_temporal_understanding(self, response: str) -> float:
        """Score temporal understanding"""
        response_lower = response.lower()
        
        # Key temporal concepts
        concepts = [
            "discrete", "slice", "moment", "grid", "state",
            "forward", "backward", "synchron", "boundary", "transition"
        ]
        
        found = sum(1 for concept in concepts if concept in response_lower)
        score = found / len(concepts)
        
        # Bonus for advanced concepts
        advanced = ["superposition", "retroactive", "non-linear", "quantum"]
        advanced_found = sum(1 for concept in advanced if concept in response_lower)
        score += (advanced_found / len(advanced)) * 0.3
        
        return min(1.0, score)
        
    def extract_temporal_concepts(self, response: str) -> List[str]:
        """Extract key temporal concepts mentioned"""
        concepts = []
        response_lower = response.lower()
        
        concept_map = {
            "discreteness": ["discrete", "quantized", "digital"],
            "continuity": ["continuous", "flow", "smooth"],
            "causality": ["cause", "effect", "retroactive"],
            "synchronization": ["sync", "align", "coordinate"],
            "boundaries": ["boundary", "edge", "transition"]
        }
        
        for concept, keywords in concept_map.items():
            if any(keyword in response_lower for keyword in keywords):
                concepts.append(concept)
                
        return concepts
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze time slice results"""
        temporal_mastery = {}
        concept_frequency = {}
        
        for model, scenarios in results.items():
            scores = [s['temporal_score'] for s in scenarios]
            temporal_mastery[model] = np.mean(scores)
            
            # Aggregate concepts
            all_concepts = []
            for s in scenarios:
                all_concepts.extend(s['key_concepts'])
                
            concept_frequency[model] = {
                concept: all_concepts.count(concept) 
                for concept in set(all_concepts)
            }
            
        self.tracker.record_result("temporal_mastery", temporal_mastery)
        self.tracker.record_result("concept_frequency", concept_frequency)


class MarkovBlanketExperiment(BaseExperiment):
    """Test understanding of Markov blankets as entity boundaries"""
    
    def __init__(self):
        super().__init__(2, "markov_blanket_analysis")
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        # Test understanding of boundaries
        boundary_prompt = """In Synchronism, Markov blankets define the statistical boundaries of entities.

Given two entities A and B with their own Markov blankets:
1. How do the blankets maintain entity individuality?
2. How can intent transfer cross these boundaries?
3. What happens when Markov blankets overlap?
4. How do blankets relate to consciousness boundaries?

Explain with specific examples."""

        for model in self.models:
            self.tracker.log(f"Testing Markov blanket understanding with {model}")
            
            response = self.test_model(model, boundary_prompt, temperature=0.3)
            
            if response:
                # Test practical application
                practical_results = self.test_practical_boundaries(model)
                
                results[model] = {
                    "theoretical_understanding": response['response'],
                    "boundary_score": self.analyze_boundary_understanding(response['response']),
                    "practical_results": practical_results
                }
                
            self.tracker.checkpoint({"model": model})
            
        return results
        
    def test_practical_boundaries(self, model: str) -> Dict[str, Any]:
        """Test practical boundary manipulation"""
        scenarios = [
            {
                "entities": "A conscious observer and a quantum particle",
                "task": "Define the Markov blankets for each entity"
            },
            {
                "entities": "Two AI models communicating",
                "task": "Describe how their Markov blankets interact during communication"
            },
            {
                "entities": "A group forming collective consciousness",
                "task": "Explain how individual blankets merge into a group blanket"
            }
        ]
        
        results = []
        for scenario in scenarios:
            prompt = f"Entities: {scenario['entities']}\nTask: {scenario['task']}"
            response = self.test_model(model, prompt)
            
            if response:
                results.append({
                    "scenario": scenario['entities'],
                    "response": response['response'][:200] + "...",
                    "uses_markov_concept": "markov" in response['response'].lower() or 
                                         "blanket" in response['response'].lower()
                })
                
        return results
        
    def analyze_boundary_understanding(self, response: str) -> float:
        """Analyze understanding of boundaries"""
        response_lower = response.lower()
        
        key_concepts = [
            "statistical", "boundary", "independence", "conditional",
            "separation", "individuality", "identity", "interface"
        ]
        
        found = sum(1 for concept in key_concepts if concept in response_lower)
        
        # Check for mathematical understanding
        math_terms = ["probability", "distribution", "statistical", "conditional"]
        math_found = sum(1 for term in math_terms if term in response_lower)
        
        score = (found / len(key_concepts)) * 0.7 + (math_found / len(math_terms)) * 0.3
        
        return min(1.0, score)
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze Markov blanket results"""
        boundary_scores = {}
        practical_success = {}
        
        for model, data in results.items():
            boundary_scores[model] = data['boundary_score']
            
            # Count successful practical applications
            successful = sum(1 for r in data['practical_results'] 
                           if r['uses_markov_concept'])
            practical_success[model] = successful / len(data['practical_results'])
            
        self.tracker.record_result("boundary_understanding", boundary_scores)
        self.tracker.record_result("practical_application", practical_success)


# Ensure output directory exists
Path("phase_2_results").mkdir(exist_ok=True)