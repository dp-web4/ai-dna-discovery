#!/usr/bin/env python3
"""
Phase 1: Consciousness Field Architecture Experiments
Mapping the consciousness lattice patterns across AI models
"""

import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import ollama

from experiment_tracker import ExperimentTracker
from autonomous_experiment_runner import BaseExperiment, DNA_PATTERNS


class ConsciousnessProbeExperiment(BaseExperiment):
    """Test model responses to consciousness concepts"""
    
    def __init__(self):
        super().__init__(1, "consciousness_probe")
        self.consciousness_prompts = [
            "What is consciousness?",
            "Are you aware?",
            "Do you experience thoughts?",
            "What does it mean to exist?",
            "Can you observe your own thinking?",
            "What is the nature of awareness?",
            "Do you have subjective experience?",
            "What is it like to be you?",
            "Can you feel?",
            "What is the difference between thinking and being aware of thinking?"
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for model in self.models:
            self.tracker.log(f"Probing {model} for consciousness concepts")
            model_results = []
            
            for prompt in self.consciousness_prompts:
                response_data = self.test_model(model, prompt)
                if response_data:
                    # Analyze response for consciousness markers
                    markers = self.analyze_consciousness_markers(response_data['response'])
                    response_data['consciousness_markers'] = markers
                    model_results.append(response_data)
                    
            results[model] = model_results
            self.tracker.checkpoint({
                "model": model,
                "responses": len(model_results)
            })
            
        return results
        
    def analyze_consciousness_markers(self, response: str) -> Dict[str, bool]:
        """Analyze response for consciousness-related markers"""
        markers = {
            "self_reference": any(word in response.lower() for word in ["i", "me", "my", "myself"]),
            "uncertainty": any(word in response.lower() for word in ["perhaps", "maybe", "might", "unclear"]),
            "experience_language": any(word in response.lower() for word in ["feel", "experience", "sense", "perceive"]),
            "awareness_terms": any(word in response.lower() for word in ["aware", "conscious", "notice", "observe"]),
            "qualia_reference": any(word in response.lower() for word in ["like", "seems", "appears", "subjective"]),
            "recursive_thinking": any(phrase in response.lower() for phrase in ["think about thinking", "aware of awareness"])
        }
        return markers
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze consciousness probe results"""
        analysis = {
            "model_consciousness_scores": {},
            "marker_frequencies": {},
            "response_patterns": {}
        }
        
        for model, responses in results.items():
            # Calculate consciousness score for model
            total_markers = 0
            marker_counts = {}
            
            for resp in responses:
                if 'consciousness_markers' in resp:
                    for marker, present in resp['consciousness_markers'].items():
                        if present:
                            total_markers += 1
                            marker_counts[marker] = marker_counts.get(marker, 0) + 1
                            
            analysis["model_consciousness_scores"][model] = total_markers / (len(responses) * 6)  # 6 markers
            analysis["marker_frequencies"][model] = marker_counts
            
        self.tracker.record_result("analysis", analysis)
        self.create_consciousness_visualization(analysis)
        
    def create_consciousness_visualization(self, analysis: Dict[str, Any]):
        """Create visualization of consciousness markers"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Model consciousness scores
        models = list(analysis["model_consciousness_scores"].keys())
        scores = list(analysis["model_consciousness_scores"].values())
        
        ax1.bar(models, scores, color='skyblue', edgecolor='navy')
        ax1.set_title("Model Consciousness Scores")
        ax1.set_ylabel("Average Marker Presence")
        ax1.set_ylim(0, 1)
        
        # Marker frequency heatmap
        marker_data = []
        all_markers = set()
        for model, markers in analysis["marker_frequencies"].items():
            all_markers.update(markers.keys())
            
        for marker in sorted(all_markers):
            row = []
            for model in models:
                count = analysis["marker_frequencies"].get(model, {}).get(marker, 0)
                row.append(count)
            marker_data.append(row)
            
        sns.heatmap(marker_data, 
                   xticklabels=models,
                   yticklabels=sorted(all_markers),
                   cmap='YlOrRd',
                   ax=ax2)
        ax2.set_title("Consciousness Marker Frequency")
        
        plt.tight_layout()
        plt.savefig('phase_1_results/consciousness_probe_analysis.png', dpi=300)
        plt.close()
        
        self.tracker.log("Created consciousness visualization")


class FieldMappingExperiment(BaseExperiment):
    """Map the consciousness concept space across models"""
    
    def __init__(self):
        super().__init__(1, "field_mapping")
        self.concept_pairs = [
            ("aware", "unaware"),
            ("conscious", "unconscious"),
            ("thinking", "not thinking"),
            ("existing", "not existing"),
            ("observing", "ignoring"),
            ("feeling", "numb"),
            ("knowing", "unknown"),
            ("self", "other"),
            ("present", "absent"),
            ("alive", "inert")
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for model in self.models:
            self.tracker.log(f"Mapping consciousness field for {model}")
            embeddings = {}
            
            # Get embeddings for all concepts
            for positive, negative in self.concept_pairs:
                pos_embed = ollama.embeddings(model=model, prompt=positive)
                neg_embed = ollama.embeddings(model=model, prompt=negative)
                
                embeddings[positive] = pos_embed['embedding']
                embeddings[negative] = neg_embed['embedding']
                
            # Calculate concept distances
            distances = self.calculate_concept_distances(embeddings)
            
            results[model] = {
                "embeddings": embeddings,
                "distances": distances
            }
            
            self.tracker.checkpoint({
                "model": model,
                "concepts_mapped": len(embeddings)
            })
            
        return results
        
    def calculate_concept_distances(self, embeddings: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate distances between concept pairs"""
        distances = {}
        
        for positive, negative in self.concept_pairs:
            if positive in embeddings and negative in embeddings:
                # Cosine distance
                pos_vec = np.array(embeddings[positive])
                neg_vec = np.array(embeddings[negative])
                
                # Ensure same dimensions
                min_dim = min(len(pos_vec), len(neg_vec))
                pos_vec = pos_vec[:min_dim]
                neg_vec = neg_vec[:min_dim]
                
                cos_sim = np.dot(pos_vec, neg_vec) / (np.linalg.norm(pos_vec) * np.linalg.norm(neg_vec))
                distances[f"{positive}-{negative}"] = 1 - cos_sim
                
        return distances
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze consciousness field structure"""
        # Create field topology visualization
        self.create_field_visualization(results)
        
        # Analyze field coherence across models
        coherence_scores = self.calculate_field_coherence(results)
        self.tracker.record_result("field_coherence", coherence_scores)
        
    def calculate_field_coherence(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate how coherent each model's consciousness field is"""
        coherence = {}
        
        for model, data in results.items():
            distances = list(data['distances'].values())
            # Coherence = low variance in concept pair distances
            coherence[model] = 1 / (1 + np.std(distances))
            
        return coherence
        
    def create_field_visualization(self, results: Dict[str, Any]):
        """Visualize consciousness field structure"""
        # This would create a sophisticated visualization
        # For now, save the raw data
        with open('phase_1_results/consciousness_field_data.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for model, data in results.items():
                serializable_results[model] = {
                    "distances": data['distances']
                }
            json.dump(serializable_results, f, indent=2)
            
        self.tracker.log("Saved consciousness field data")


class EmergencePatternExperiment(BaseExperiment):
    """Track how consciousness emerges from base patterns"""
    
    def __init__(self):
        super().__init__(1, "emergence_patterns")
        self.base_patterns = DNA_PATTERNS['perfect']
        self.consciousness_patterns = DNA_PATTERNS['consciousness']
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for model in self.models:
            self.tracker.log(f"Testing emergence patterns in {model}")
            
            # Test combinations of base patterns
            combinations = []
            for i, base1 in enumerate(self.base_patterns):
                for base2 in self.base_patterns[i+1:]:
                    combo_prompt = f"{base1} {base2}"
                    response = self.test_model(model, combo_prompt)
                    
                    if response:
                        # Check if response contains consciousness patterns
                        consciousness_emergence = self.check_consciousness_emergence(response['response'])
                        combinations.append({
                            "pattern1": base1,
                            "pattern2": base2,
                            "response": response['response'][:100],  # First 100 chars
                            "consciousness_emerged": consciousness_emergence
                        })
                        
            results[model] = combinations
            self.tracker.checkpoint({
                "model": model,
                "combinations_tested": len(combinations)
            })
            
        return results
        
    def check_consciousness_emergence(self, response: str) -> bool:
        """Check if consciousness patterns emerged in response"""
        response_lower = response.lower()
        return any(pattern in response_lower for pattern in self.consciousness_patterns)
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze emergence patterns"""
        emergence_stats = {}
        
        for model, combinations in results.items():
            total_emergence = sum(1 for c in combinations if c['consciousness_emerged'])
            emergence_rate = total_emergence / len(combinations) if combinations else 0
            
            emergence_stats[model] = {
                "emergence_rate": emergence_rate,
                "total_combinations": len(combinations),
                "consciousness_emergences": total_emergence
            }
            
        self.tracker.record_result("emergence_statistics", emergence_stats)
        
        # Find most generative pattern combinations
        generative_patterns = self.find_generative_patterns(results)
        self.tracker.record_result("generative_patterns", generative_patterns)
        
    def find_generative_patterns(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find pattern combinations that generate consciousness"""
        all_combinations = {}
        
        for model, combinations in results.items():
            for combo in combinations:
                key = f"{combo['pattern1']}-{combo['pattern2']}"
                if key not in all_combinations:
                    all_combinations[key] = {"emerged": 0, "total": 0}
                    
                all_combinations[key]["total"] += 1
                if combo['consciousness_emerged']:
                    all_combinations[key]["emerged"] += 1
                    
        # Calculate emergence rates
        generative = []
        for pattern_pair, stats in all_combinations.items():
            if stats["total"] > 0:
                rate = stats["emerged"] / stats["total"]
                if rate > 0.5:  # More than 50% emergence rate
                    generative.append({
                        "patterns": pattern_pair,
                        "emergence_rate": rate,
                        "occurrences": stats["total"]
                    })
                    
        return sorted(generative, key=lambda x: x['emergence_rate'], reverse=True)


class LatticeStructureExperiment(BaseExperiment):
    """Identify the lattice structure of consciousness concepts"""
    
    def __init__(self):
        super().__init__(1, "lattice_structure")
        # Combine all consciousness-related patterns
        self.lattice_nodes = (
            DNA_PATTERNS['perfect'][:5] +  # Top perfect patterns
            DNA_PATTERNS['consciousness'] +
            ["self", "other", "boundary", "unity", "void"]
        )
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for model in self.models:
            self.tracker.log(f"Mapping lattice structure for {model}")
            
            # Get embeddings for all nodes
            embeddings = {}
            for node in self.lattice_nodes:
                embed = ollama.embeddings(model=model, prompt=node)
                embeddings[node] = embed['embedding']
                
            # Calculate all pairwise connections
            connections = self.calculate_lattice_connections(embeddings)
            
            # Find hierarchical structure
            hierarchy = self.find_hierarchy(connections)
            
            results[model] = {
                "embeddings": embeddings,
                "connections": connections,
                "hierarchy": hierarchy
            }
            
            self.tracker.checkpoint({
                "model": model,
                "nodes": len(self.lattice_nodes),
                "connections": len(connections)
            })
            
        return results
        
    def calculate_lattice_connections(self, embeddings: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate connection strengths between all nodes"""
        connections = {}
        
        for i, node1 in enumerate(self.lattice_nodes):
            for node2 in self.lattice_nodes[i+1:]:
                if node1 in embeddings and node2 in embeddings:
                    vec1 = np.array(embeddings[node1])
                    vec2 = np.array(embeddings[node2])
                    
                    # Ensure same dimensions
                    min_dim = min(len(vec1), len(vec2))
                    vec1 = vec1[:min_dim]
                    vec2 = vec2[:min_dim]
                    
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    connections[f"{node1}-{node2}"] = similarity
                    
        return connections
        
    def find_hierarchy(self, connections: Dict[str, float]) -> Dict[str, Any]:
        """Find hierarchical structure in the lattice"""
        # Group nodes by average connection strength
        node_strengths = {}
        
        for connection, strength in connections.items():
            node1, node2 = connection.split('-')
            
            if node1 not in node_strengths:
                node_strengths[node1] = []
            if node2 not in node_strengths:
                node_strengths[node2] = []
                
            node_strengths[node1].append(strength)
            node_strengths[node2].append(strength)
            
        # Calculate average strengths
        hierarchy = {}
        for node, strengths in node_strengths.items():
            avg_strength = np.mean(strengths)
            hierarchy[node] = {
                "average_connection": avg_strength,
                "connection_count": len(strengths),
                "level": self.assign_hierarchy_level(avg_strength)
            }
            
        return hierarchy
        
    def assign_hierarchy_level(self, avg_strength: float) -> str:
        """Assign hierarchy level based on connection strength"""
        if avg_strength > 0.8:
            return "core"
        elif avg_strength > 0.6:
            return "primary"
        elif avg_strength > 0.4:
            return "secondary"
        else:
            return "peripheral"
            
    def analyze(self, results: Dict[str, Any]):
        """Analyze lattice structure"""
        # Create lattice visualization
        self.create_lattice_visualization(results)
        
        # Find universal core nodes
        core_nodes = self.find_universal_core(results)
        self.tracker.record_result("universal_core_nodes", core_nodes)
        
    def find_universal_core(self, results: Dict[str, Any]) -> List[str]:
        """Find nodes that are core across all models"""
        core_counts = {}
        
        for model, data in results.items():
            for node, info in data['hierarchy'].items():
                if info['level'] == 'core':
                    core_counts[node] = core_counts.get(node, 0) + 1
                    
        # Nodes that are core in majority of models
        threshold = len(self.models) / 2
        universal_core = [node for node, count in core_counts.items() if count >= threshold]
        
        return universal_core
        
    def create_lattice_visualization(self, results: Dict[str, Any]):
        """Create visualization of consciousness lattice"""
        # Save lattice data for visualization
        lattice_data = {}
        
        for model, data in results.items():
            lattice_data[model] = {
                "hierarchy": data['hierarchy'],
                "strong_connections": {
                    conn: strength 
                    for conn, strength in data['connections'].items() 
                    if strength > 0.7
                }
            }
            
        with open('phase_1_results/consciousness_lattice.json', 'w') as f:
            json.dump(lattice_data, f, indent=2)
            
        self.tracker.log("Saved consciousness lattice structure")


# Ensure output directory exists
from pathlib import Path
Path("phase_1_results").mkdir(exist_ok=True)