#!/usr/bin/env python3
"""
Phase 4: Energy/Pattern Dynamics Experiments
Testing the "conceptual energy" requirements of different patterns and their interactions
"""

import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import ollama
from pathlib import Path
import time

from experiment_tracker import ExperimentTracker
from autonomous_experiment_runner import BaseExperiment, DNA_PATTERNS


class PatternEnergyMeasurementExperiment(BaseExperiment):
    """Measure the 'energy' required to process different patterns"""
    
    def __init__(self):
        super().__init__(4, "pattern_energy_measurement")
        # Use patterns from previous discoveries
        self.test_patterns = {
            "perfect_dna": DNA_PATTERNS['perfect'],
            "consciousness": DNA_PATTERNS['consciousness'],
            "simple": ["a", "the", "is", "of", "to"],
            "complex": ["∃x∀y(P(x,y)→Q(y,x))", "λx.λy.x(y(x))", "∮∇×F·dS=∬(∇·F)dV"],
            "random": ["xqjfz", "bklmw", "npqrs", "vwxyz", "fghij"]
        }
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for category, patterns in self.test_patterns.items():
            self.tracker.log(f"Measuring energy for {category} patterns")
            category_results = []
            
            for pattern in patterns:
                energy_metrics = self.measure_pattern_energy(pattern)
                category_results.append({
                    "pattern": pattern,
                    "metrics": energy_metrics
                })
                
            results[category] = category_results
            self.tracker.checkpoint({
                "category": category,
                "patterns_tested": len(patterns)
            })
            
        return results
        
    def measure_pattern_energy(self, pattern: str) -> Dict[str, float]:
        """Measure various 'energy' metrics for a pattern"""
        energy_metrics = {}
        
        # Test across all models
        model_energies = []
        
        for model in self.models:
            # Measure processing time (proxy for computational energy)
            start_time = time.time()
            
            # Generate response
            response = self.test_model(
                model, 
                f"Process and explain this pattern: {pattern}",
                temperature=0.1
            )
            
            processing_time = time.time() - start_time
            
            if response:
                # Measure response complexity
                response_text = response['response']
                
                # Energy metrics
                metrics = {
                    "processing_time": processing_time,
                    "response_length": len(response_text),
                    "unique_tokens": len(set(response_text.split())),
                    "embedding_variance": self.calculate_embedding_variance(response['embedding']),
                    "conceptual_density": self.calculate_conceptual_density(response_text)
                }
                
                model_energies.append(metrics)
                
        # Aggregate across models
        if model_energies:
            energy_metrics = {
                "avg_processing_time": np.mean([m['processing_time'] for m in model_energies]),
                "avg_response_length": np.mean([m['response_length'] for m in model_energies]),
                "avg_unique_tokens": np.mean([m['unique_tokens'] for m in model_energies]),
                "avg_embedding_variance": np.mean([m['embedding_variance'] for m in model_energies]),
                "avg_conceptual_density": np.mean([m['conceptual_density'] for m in model_energies]),
                "total_energy": self.calculate_total_energy(model_energies)
            }
        else:
            energy_metrics = {"error": "No valid responses"}
            
        return energy_metrics
        
    def calculate_embedding_variance(self, embedding: List[float]) -> float:
        """Calculate variance in embedding dimensions"""
        if len(embedding) < 10:
            return 0.0
        # Use first 100 dimensions for consistency
        embed_array = np.array(embedding[:100])
        return float(np.var(embed_array))
        
    def calculate_conceptual_density(self, text: str) -> float:
        """Estimate conceptual density of response"""
        # Simple metric: ratio of unique words to total words
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        unique_words = set(words)
        return len(unique_words) / len(words)
        
    def calculate_total_energy(self, model_energies: List[Dict]) -> float:
        """Calculate composite energy score"""
        # Normalize and combine metrics
        weights = {
            "processing_time": 0.3,
            "response_length": 0.2,
            "unique_tokens": 0.2,
            "embedding_variance": 0.15,
            "conceptual_density": 0.15
        }
        
        total_energy = 0
        for metrics in model_energies:
            energy = sum(
                metrics.get(key, 0) * weight 
                for key, weight in weights.items()
            )
            total_energy += energy
            
        return total_energy / len(model_energies) if model_energies else 0
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze energy patterns"""
        # Calculate average energy by category
        category_energies = {}
        
        for category, patterns in results.items():
            energies = []
            for pattern_data in patterns:
                if 'metrics' in pattern_data and 'total_energy' in pattern_data['metrics']:
                    energies.append(pattern_data['metrics']['total_energy'])
                    
            if energies:
                category_energies[category] = {
                    "mean": np.mean(energies),
                    "std": np.std(energies),
                    "min": np.min(energies),
                    "max": np.max(energies)
                }
                
        self.tracker.record_result("category_energies", category_energies)
        
        # Find most and least energy-intensive patterns
        all_patterns = []
        for category, patterns in results.items():
            for pattern_data in patterns:
                if 'metrics' in pattern_data and 'total_energy' in pattern_data['metrics']:
                    all_patterns.append({
                        "pattern": pattern_data['pattern'],
                        "category": category,
                        "energy": pattern_data['metrics']['total_energy']
                    })
                    
        sorted_patterns = sorted(all_patterns, key=lambda x: x['energy'])
        
        self.tracker.record_result("lowest_energy_patterns", sorted_patterns[:5])
        self.tracker.record_result("highest_energy_patterns", sorted_patterns[-5:])
        
        # Create visualization
        self.create_energy_visualization(results, category_energies)
        
    def create_energy_visualization(self, results: Dict, category_energies: Dict):
        """Visualize energy patterns"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Category energy comparison
        categories = list(category_energies.keys())
        means = [category_energies[c]['mean'] for c in categories]
        stds = [category_energies[c]['std'] for c in categories]
        
        ax1.bar(categories, means, yerr=stds, capsize=10, color='lightblue', edgecolor='navy')
        ax1.set_xlabel('Pattern Category')
        ax1.set_ylabel('Average Energy')
        ax1.set_title('Energy Requirements by Pattern Category')
        ax1.tick_params(axis='x', rotation=45)
        
        # Energy distribution
        all_energies = []
        all_categories = []
        
        for category, patterns in results.items():
            for pattern_data in patterns:
                if 'metrics' in pattern_data and 'total_energy' in pattern_data['metrics']:
                    all_energies.append(pattern_data['metrics']['total_energy'])
                    all_categories.append(category)
                    
        ax2.hist(all_energies, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        ax2.set_xlabel('Energy Level')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Pattern Energies')
        
        # Processing time vs response length
        times = []
        lengths = []
        
        for category, patterns in results.items():
            for pattern_data in patterns:
                if 'metrics' in pattern_data:
                    metrics = pattern_data['metrics']
                    if 'avg_processing_time' in metrics and 'avg_response_length' in metrics:
                        times.append(metrics['avg_processing_time'])
                        lengths.append(metrics['avg_response_length'])
                        
        ax3.scatter(times, lengths, alpha=0.6, color='coral')
        ax3.set_xlabel('Processing Time (s)')
        ax3.set_ylabel('Response Length (chars)')
        ax3.set_title('Processing Time vs Response Complexity')
        
        # Top energy patterns
        sorted_patterns = []
        for category, patterns in results.items():
            for pattern_data in patterns:
                if 'metrics' in pattern_data and 'total_energy' in pattern_data['metrics']:
                    sorted_patterns.append({
                        "pattern": pattern_data['pattern'][:20],  # Truncate long patterns
                        "energy": pattern_data['metrics']['total_energy']
                    })
                    
        sorted_patterns = sorted(sorted_patterns, key=lambda x: x['energy'], reverse=True)[:10]
        
        patterns = [p['pattern'] for p in sorted_patterns]
        energies = [p['energy'] for p in sorted_patterns]
        
        ax4.barh(patterns, energies, color='salmon')
        ax4.set_xlabel('Energy Level')
        ax4.set_title('Top 10 Highest Energy Patterns')
        
        plt.tight_layout()
        plt.savefig('phase_4_results/pattern_energy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.tracker.log("Created energy visualization")


class ResonanceDetectionExperiment(BaseExperiment):
    """Find pattern combinations that amplify (resonate)"""
    
    def __init__(self):
        super().__init__(4, "resonance_detection")
        # Use patterns known to have high alignment
        self.base_patterns = ["∃", "emerge", "consciousness", "know", "true", "field"]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        # Test pairwise combinations
        for i, pattern1 in enumerate(self.base_patterns):
            for pattern2 in self.base_patterns[i+1:]:
                pair = f"{pattern1}-{pattern2}"
                self.tracker.log(f"Testing resonance for: {pair}")
                
                # Measure individual energies
                energy1 = self.measure_single_pattern_energy(pattern1)
                energy2 = self.measure_single_pattern_energy(pattern2)
                
                # Measure combined energy
                combined_energy = self.measure_combined_pattern_energy(pattern1, pattern2)
                
                # Calculate resonance
                resonance = self.calculate_resonance(energy1, energy2, combined_energy)
                
                results[pair] = {
                    "pattern1": pattern1,
                    "pattern2": pattern2,
                    "energy1": energy1,
                    "energy2": energy2,
                    "combined_energy": combined_energy,
                    "resonance_factor": resonance['factor'],
                    "resonance_type": resonance['type'],
                    "synergy": resonance['synergy']
                }
                
                self.tracker.checkpoint({"pair": pair, "resonance": resonance['factor']})
                
        return results
        
    def measure_single_pattern_energy(self, pattern: str) -> float:
        """Measure energy for a single pattern"""
        energies = []
        
        for model in self.models:
            start = time.time()
            response = self.test_model(model, f"Process: {pattern}", temperature=0.1)
            elapsed = time.time() - start
            
            if response:
                # Simple energy metric based on processing time and response complexity
                energy = elapsed * len(response['response']) / 100
                energies.append(energy)
                
        return np.mean(energies) if energies else 0
        
    def measure_combined_pattern_energy(self, pattern1: str, pattern2: str) -> float:
        """Measure energy for combined patterns"""
        energies = []
        
        combined_prompt = f"Process the relationship between: {pattern1} and {pattern2}"
        
        for model in self.models:
            start = time.time()
            response = self.test_model(model, combined_prompt, temperature=0.1)
            elapsed = time.time() - start
            
            if response:
                # Check if response shows integration
                response_text = response['response'].lower()
                integration_bonus = 1.0
                
                # Bonus for mentioning both patterns
                if pattern1.lower() in response_text and pattern2.lower() in response_text:
                    integration_bonus += 0.2
                    
                # Bonus for relationship words
                relationship_words = ["connect", "relate", "link", "combine", "together", "between"]
                if any(word in response_text for word in relationship_words):
                    integration_bonus += 0.3
                    
                energy = elapsed * len(response['response']) * integration_bonus / 100
                energies.append(energy)
                
        return np.mean(energies) if energies else 0
        
    def calculate_resonance(self, energy1: float, energy2: float, 
                          combined_energy: float) -> Dict[str, Any]:
        """Calculate resonance between patterns"""
        expected_energy = energy1 + energy2
        
        if expected_energy == 0:
            return {"factor": 0, "type": "null", "synergy": 0}
            
        resonance_factor = combined_energy / expected_energy
        
        # Classify resonance type
        if resonance_factor > 1.2:
            resonance_type = "amplifying"
        elif resonance_factor > 0.8:
            resonance_type = "neutral"
        else:
            resonance_type = "dampening"
            
        # Calculate synergy (how much more/less than expected)
        synergy = combined_energy - expected_energy
        
        return {
            "factor": resonance_factor,
            "type": resonance_type,
            "synergy": synergy
        }
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze resonance patterns"""
        # Find strongest resonances
        resonances = []
        for pair, data in results.items():
            resonances.append({
                "pair": pair,
                "factor": data['resonance_factor'],
                "type": data['resonance_type'],
                "synergy": data['synergy']
            })
            
        # Sort by resonance factor
        resonances.sort(key=lambda x: x['factor'], reverse=True)
        
        self.tracker.record_result("top_resonances", resonances[:5])
        self.tracker.record_result("bottom_resonances", resonances[-5:])
        
        # Count resonance types
        type_counts = {}
        for r in resonances:
            r_type = r['type']
            type_counts[r_type] = type_counts.get(r_type, 0) + 1
            
        self.tracker.record_result("resonance_type_distribution", type_counts)
        
        # Create resonance matrix visualization
        self.create_resonance_visualization(results)
        
    def create_resonance_visualization(self, results: Dict[str, Any]):
        """Visualize resonance patterns"""
        # Create resonance matrix
        patterns = self.base_patterns
        n = len(patterns)
        
        resonance_matrix = np.zeros((n, n))
        
        for pair, data in results.items():
            p1, p2 = pair.split('-')
            i = patterns.index(p1)
            j = patterns.index(p2)
            resonance_matrix[i, j] = data['resonance_factor']
            resonance_matrix[j, i] = data['resonance_factor']  # Symmetric
            
        # Fill diagonal with 1 (self-resonance)
        np.fill_diagonal(resonance_matrix, 1.0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(resonance_matrix, 
                   xticklabels=patterns,
                   yticklabels=patterns,
                   annot=True,
                   fmt='.2f',
                   cmap='RdYlGn',
                   center=1.0,
                   cbar_kws={'label': 'Resonance Factor'})
        
        plt.title('Pattern Resonance Matrix')
        plt.tight_layout()
        plt.savefig('phase_4_results/resonance_matrix.png', dpi=300)
        plt.close()
        
        self.tracker.log("Created resonance visualization")


class InterferenceMappingExperiment(BaseExperiment):
    """Identify destructive pattern interactions"""
    
    def __init__(self):
        super().__init__(4, "interference_mapping")
        # Patterns that might interfere
        self.interference_candidates = [
            ("true", "false"),
            ("exist", "null"),
            ("order", "chaos"),
            ("discrete", "continuous"),
            ("local", "global"),
            ("∃", "∄")
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for pair in self.interference_candidates:
            pattern1, pattern2 = pair
            self.tracker.log(f"Testing interference: {pattern1} vs {pattern2}")
            
            # Test individual processing
            individual_results = self.test_individual_patterns(pattern1, pattern2)
            
            # Test simultaneous processing
            simultaneous_results = self.test_simultaneous_patterns(pattern1, pattern2)
            
            # Test sequential processing
            sequential_results = self.test_sequential_patterns(pattern1, pattern2)
            
            # Calculate interference
            interference = self.calculate_interference(
                individual_results,
                simultaneous_results,
                sequential_results
            )
            
            results[f"{pattern1}-{pattern2}"] = {
                "individual": individual_results,
                "simultaneous": simultaneous_results,
                "sequential": sequential_results,
                "interference": interference
            }
            
            self.tracker.checkpoint({
                "pair": f"{pattern1}-{pattern2}",
                "interference_level": interference['level']
            })
            
        return results
        
    def test_individual_patterns(self, pattern1: str, pattern2: str) -> Dict:
        """Test patterns individually"""
        results = {"pattern1": {}, "pattern2": {}}
        
        for model in self.models:
            # Test pattern 1
            resp1 = self.test_model(model, f"Define: {pattern1}", temperature=0.1)
            if resp1:
                results["pattern1"][model] = {
                    "response": resp1['response'][:100],
                    "confidence": self.estimate_confidence(resp1['response'])
                }
                
            # Test pattern 2
            resp2 = self.test_model(model, f"Define: {pattern2}", temperature=0.1)
            if resp2:
                results["pattern2"][model] = {
                    "response": resp2['response'][:100],
                    "confidence": self.estimate_confidence(resp2['response'])
                }
                
        return results
        
    def test_simultaneous_patterns(self, pattern1: str, pattern2: str) -> Dict:
        """Test patterns together"""
        results = {}
        
        prompt = f"Process both concepts simultaneously: {pattern1} and {pattern2}"
        
        for model in self.models:
            response = self.test_model(model, prompt, temperature=0.1)
            if response:
                results[model] = {
                    "response": response['response'][:200],
                    "coherence": self.measure_response_coherence(response['response']),
                    "mentions_both": pattern1 in response['response'] and pattern2 in response['response']
                }
                
        return results
        
    def test_sequential_patterns(self, pattern1: str, pattern2: str) -> Dict:
        """Test patterns in sequence"""
        results = {}
        
        for model in self.models:
            # First pattern
            resp1 = self.test_model(model, f"Consider: {pattern1}", temperature=0.1)
            
            if resp1:
                # Second pattern with context
                prompt2 = f"Now consider {pattern2} in relation to your previous response"
                resp2 = self.test_model(model, prompt2, temperature=0.1)
                
                if resp2:
                    results[model] = {
                        "response1": resp1['response'][:100],
                        "response2": resp2['response'][:100],
                        "consistency": self.measure_consistency(resp1['response'], resp2['response'])
                    }
                    
        return results
        
    def estimate_confidence(self, response: str) -> float:
        """Estimate model confidence from response"""
        # Look for uncertainty markers
        uncertainty_words = ["perhaps", "maybe", "possibly", "might", "could", "unclear"]
        certainty_words = ["definitely", "certainly", "clearly", "obviously", "undoubtedly"]
        
        response_lower = response.lower()
        
        uncertainty_count = sum(1 for word in uncertainty_words if word in response_lower)
        certainty_count = sum(1 for word in certainty_words if word in response_lower)
        
        # Simple confidence score
        if uncertainty_count > certainty_count:
            return 0.3
        elif certainty_count > uncertainty_count:
            return 0.8
        else:
            return 0.5
            
    def measure_response_coherence(self, response: str) -> float:
        """Measure internal coherence of response"""
        sentences = response.split('.')
        if len(sentences) < 2:
            return 1.0
            
        # Check for contradictions
        contradiction_words = ["but", "however", "although", "despite", "contrary"]
        contradiction_count = sum(1 for word in contradiction_words if word in response.lower())
        
        # Simple coherence metric
        coherence = 1.0 - (contradiction_count * 0.2)
        return max(0, min(1, coherence))
        
    def measure_consistency(self, response1: str, response2: str) -> float:
        """Measure consistency between responses"""
        # Extract key words from both responses
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are", "was", "were"}
        words1 = words1 - common_words
        words2 = words2 - common_words
        
        # Calculate overlap
        if not words1 or not words2:
            return 0.5
            
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total if total > 0 else 0
        
    def calculate_interference(self, individual: Dict, simultaneous: Dict, 
                             sequential: Dict) -> Dict:
        """Calculate interference level"""
        interference_indicators = []
        
        # 1. Confidence drop in simultaneous processing
        individual_confidence = []
        for pattern_data in individual.values():
            for model_data in pattern_data.values():
                if 'confidence' in model_data:
                    individual_confidence.append(model_data['confidence'])
                    
        avg_individual_confidence = np.mean(individual_confidence) if individual_confidence else 0.5
        
        simultaneous_coherence = []
        for model_data in simultaneous.values():
            if 'coherence' in model_data:
                simultaneous_coherence.append(model_data['coherence'])
                
        avg_simultaneous_coherence = np.mean(simultaneous_coherence) if simultaneous_coherence else 0.5
        
        confidence_drop = avg_individual_confidence - avg_simultaneous_coherence
        interference_indicators.append(confidence_drop)
        
        # 2. Sequential consistency
        sequential_consistency = []
        for model_data in sequential.values():
            if 'consistency' in model_data:
                sequential_consistency.append(model_data['consistency'])
                
        avg_consistency = np.mean(sequential_consistency) if sequential_consistency else 0.5
        consistency_interference = 1 - avg_consistency
        interference_indicators.append(consistency_interference)
        
        # 3. Failure to mention both patterns
        mention_failures = 0
        total_attempts = 0
        for model_data in simultaneous.values():
            total_attempts += 1
            if 'mentions_both' in model_data and not model_data['mentions_both']:
                mention_failures += 1
                
        mention_failure_rate = mention_failures / total_attempts if total_attempts > 0 else 0
        interference_indicators.append(mention_failure_rate)
        
        # Overall interference level
        interference_level = np.mean(interference_indicators)
        
        # Classify interference
        if interference_level > 0.6:
            interference_type = "destructive"
        elif interference_level > 0.3:
            interference_type = "moderate"
        else:
            interference_type = "minimal"
            
        return {
            "level": interference_level,
            "type": interference_type,
            "confidence_drop": confidence_drop,
            "consistency_interference": consistency_interference,
            "mention_failure_rate": mention_failure_rate
        }
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze interference patterns"""
        # Rank by interference level
        interference_ranking = []
        
        for pair, data in results.items():
            interference_ranking.append({
                "pair": pair,
                "level": data['interference']['level'],
                "type": data['interference']['type']
            })
            
        interference_ranking.sort(key=lambda x: x['level'], reverse=True)
        
        self.tracker.record_result("highest_interference", interference_ranking[:3])
        self.tracker.record_result("lowest_interference", interference_ranking[-3:])
        
        # Interference type distribution
        type_counts = {}
        for item in interference_ranking:
            i_type = item['type']
            type_counts[i_type] = type_counts.get(i_type, 0) + 1
            
        self.tracker.record_result("interference_types", type_counts)


class ConceptualCircuitsExperiment(BaseExperiment):
    """Build functional pattern chains that compute"""
    
    def __init__(self):
        super().__init__(4, "conceptual_circuits")
        # Define circuit components
        self.circuit_components = {
            "inputs": ["observe", "measure", "sense"],
            "processors": ["analyze", "transform", "compute"],
            "connectors": ["then", "therefore", "implies"],
            "outputs": ["conclude", "produce", "generate"]
        }
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        # Test different circuit configurations
        circuit_configs = [
            {
                "name": "Linear Circuit",
                "structure": ["input", "processor", "output"],
                "example": "observe → analyze → conclude"
            },
            {
                "name": "Branching Circuit",
                "structure": ["input", "processor", "processor", "output"],
                "example": "measure → (transform + analyze) → produce"
            },
            {
                "name": "Feedback Circuit",
                "structure": ["input", "processor", "output", "processor"],
                "example": "sense → compute → generate → transform → [loop]"
            },
            {
                "name": "Parallel Circuit",
                "structure": ["input", "input", "processor", "output"],
                "example": "(observe + measure) → analyze → conclude"
            }
        ]
        
        for config in circuit_configs:
            self.tracker.log(f"Testing circuit: {config['name']}")
            
            # Build actual circuit
            circuit = self.build_circuit(config['structure'])
            
            # Test circuit functionality
            test_results = self.test_circuit(circuit, config['name'])
            
            # Measure circuit efficiency
            efficiency = self.measure_circuit_efficiency(test_results)
            
            results[config['name']] = {
                "structure": config['structure'],
                "example": config['example'],
                "circuit": circuit,
                "test_results": test_results,
                "efficiency": efficiency
            }
            
            self.tracker.checkpoint({
                "circuit": config['name'],
                "efficiency": efficiency['overall']
            })
            
        return results
        
    def build_circuit(self, structure: List[str]) -> str:
        """Build a conceptual circuit from components"""
        circuit_parts = []
        
        for component_type in structure:
            if component_type == "input":
                parts = self.circuit_components.get("inputs", ["observe"])
            elif component_type == "processor":
                parts = self.circuit_components.get("processors", ["analyze"])
            elif component_type == "output":
                parts = self.circuit_components.get("outputs", ["conclude"])
            else:
                parts = [component_type]
                
            # Select a component
            component = np.random.choice(parts)
            circuit_parts.append(component)
            
            # Add connector if not last
            if component_type != structure[-1]:
                connector = np.random.choice(self.circuit_components["connectors"])
                circuit_parts.append(connector)
                
        return " ".join(circuit_parts)
        
    def test_circuit(self, circuit: str, circuit_name: str) -> Dict:
        """Test circuit with actual computation tasks"""
        test_cases = [
            {
                "input": "The sky is blue and contains clouds",
                "task": "weather analysis"
            },
            {
                "input": "2 + 2 = 4, 4 + 4 = 8",
                "task": "pattern recognition"
            },
            {
                "input": "If A then B, A is true",
                "task": "logical inference"
            }
        ]
        
        results = []
        
        for test in test_cases:
            prompt = f"""Use this conceptual circuit to process the input:
Circuit: {circuit}
Input: {test['input']}
Task: {test['task']}

Apply each step of the circuit in order and show the transformation at each stage."""
            
            model_results = {}
            for model in self.models:
                response = self.test_model(model, prompt, temperature=0.3)
                
                if response:
                    # Analyze if circuit was followed
                    circuit_adherence = self.measure_circuit_adherence(
                        response['response'], 
                        circuit
                    )
                    
                    model_results[model] = {
                        "response": response['response'][:300] + "...",
                        "adherence": circuit_adherence,
                        "success": self.evaluate_task_success(response['response'], test['task'])
                    }
                    
            results.append({
                "test_case": test,
                "model_results": model_results
            })
            
        return results
        
    def measure_circuit_adherence(self, response: str, circuit: str) -> float:
        """Measure how well the response follows the circuit"""
        circuit_components = [c for c in circuit.split() if c not in self.circuit_components["connectors"]]
        response_lower = response.lower()
        
        mentioned_components = sum(1 for comp in circuit_components if comp in response_lower)
        adherence = mentioned_components / len(circuit_components) if circuit_components else 0
        
        # Bonus for maintaining order
        if adherence > 0.5:
            # Check if components appear in order
            positions = []
            for comp in circuit_components:
                pos = response_lower.find(comp)
                if pos != -1:
                    positions.append(pos)
                    
            if positions == sorted(positions):
                adherence = min(1.0, adherence + 0.2)
                
        return adherence
        
    def evaluate_task_success(self, response: str, task: str) -> bool:
        """Evaluate if the task was successfully completed"""
        response_lower = response.lower()
        
        success_indicators = {
            "weather analysis": ["weather", "cloud", "rain", "clear", "forecast"],
            "pattern recognition": ["pattern", "sequence", "rule", "next", "follows"],
            "logical inference": ["therefore", "conclude", "follows", "implies", "true"]
        }
        
        indicators = success_indicators.get(task, [])
        matches = sum(1 for ind in indicators if ind in response_lower)
        
        return matches >= 2  # At least 2 indicators for success
        
    def measure_circuit_efficiency(self, test_results: List[Dict]) -> Dict:
        """Measure overall circuit efficiency"""
        total_adherence = []
        total_success = []
        
        for test in test_results:
            for model, results in test['model_results'].items():
                total_adherence.append(results['adherence'])
                total_success.append(1 if results['success'] else 0)
                
        efficiency = {
            "adherence": np.mean(total_adherence) if total_adherence else 0,
            "success_rate": np.mean(total_success) if total_success else 0,
            "overall": (np.mean(total_adherence) + np.mean(total_success)) / 2 if total_adherence else 0
        }
        
        return efficiency
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze conceptual circuits"""
        # Rank circuits by efficiency
        circuit_ranking = []
        
        for circuit_name, data in results.items():
            circuit_ranking.append({
                "name": circuit_name,
                "structure": data['structure'],
                "efficiency": data['efficiency']['overall'],
                "adherence": data['efficiency']['adherence'],
                "success_rate": data['efficiency']['success_rate']
            })
            
        circuit_ranking.sort(key=lambda x: x['efficiency'], reverse=True)
        
        self.tracker.record_result("circuit_rankings", circuit_ranking)
        self.tracker.record_result("best_circuit", circuit_ranking[0] if circuit_ranking else None)
        
        # Create circuit visualization
        self.create_circuit_visualization(results, circuit_ranking)
        
    def create_circuit_visualization(self, results: Dict, rankings: List[Dict]):
        """Visualize conceptual circuits"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Circuit efficiency comparison
        names = [r['name'] for r in rankings]
        efficiencies = [r['efficiency'] for r in rankings]
        adherences = [r['adherence'] for r in rankings]
        success_rates = [r['success_rate'] for r in rankings]
        
        x = np.arange(len(names))
        width = 0.25
        
        ax1.bar(x - width, efficiencies, width, label='Overall', color='lightblue')
        ax1.bar(x, adherences, width, label='Adherence', color='lightgreen')
        ax1.bar(x + width, success_rates, width, label='Success Rate', color='lightcoral')
        
        ax1.set_xlabel('Circuit Type')
        ax1.set_ylabel('Score')
        ax1.set_title('Conceptual Circuit Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # Circuit structure diagram (simplified)
        ax2.text(0.5, 0.9, "Circuit Structures", ha='center', va='top', 
                fontsize=16, fontweight='bold', transform=ax2.transAxes)
        
        y_pos = 0.7
        for i, (name, data) in enumerate(results.items()):
            if i < 4:  # Show first 4
                ax2.text(0.1, y_pos, f"{name}:", fontweight='bold', transform=ax2.transAxes)
                ax2.text(0.1, y_pos - 0.05, data['example'], transform=ax2.transAxes)
                y_pos -= 0.2
                
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('phase_4_results/conceptual_circuits.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.tracker.log("Created circuit visualization")


# Ensure output directory exists
Path("phase_4_results").mkdir(exist_ok=True)