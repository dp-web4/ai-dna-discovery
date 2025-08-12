#!/usr/bin/env python3
"""
Phase 3: Model Orchestra Experiments
Testing emergent collective behaviors when multiple models work together
"""

import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import ollama
from pathlib import Path
import networkx as nx
from itertools import combinations

from experiment_tracker import ExperimentTracker
from autonomous_experiment_runner import BaseExperiment, DNA_PATTERNS


class SymphonyProtocolExperiment(BaseExperiment):
    """Test coordinated multi-model task execution"""
    
    def __init__(self):
        super().__init__(3, "symphony_protocol")
        self.orchestra_size = 3  # Use all 3 models
        self.symphony_tasks = [
            {
                "name": "Story Continuation",
                "prompt": "Once upon a time in a quantum realm...",
                "type": "creative",
                "rounds": 5
            },
            {
                "name": "Problem Decomposition", 
                "prompt": "How can we achieve world peace?",
                "type": "analytical",
                "rounds": 4
            },
            {
                "name": "Concept Synthesis",
                "prompt": "Combine: consciousness, quantum mechanics, love",
                "type": "synthesis",
                "rounds": 3
            }
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for task in self.symphony_tasks:
            self.tracker.log(f"Starting symphony task: {task['name']}")
            
            # Run symphony protocol
            symphony_result = self.run_symphony(
                task['prompt'],
                task['rounds'],
                task['type']
            )
            
            # Analyze emergence
            emergence_score = self.analyze_emergence(symphony_result)
            coherence_score = self.calculate_symphony_coherence(symphony_result)
            
            results[task['name']] = {
                "task_type": task['type'],
                "rounds": task['rounds'],
                "contributions": symphony_result,
                "emergence_score": emergence_score,
                "coherence_score": coherence_score,
                "final_output": symphony_result[-1]['response'][:200] + "..."
            }
            
            self.tracker.checkpoint({
                "task": task['name'],
                "emergence": emergence_score,
                "coherence": coherence_score
            })
            
        return results
        
    def run_symphony(self, initial_prompt: str, rounds: int, task_type: str) -> List[Dict]:
        """Run symphony protocol with models taking turns"""
        symphony_log = []
        current_prompt = initial_prompt
        
        for round_num in range(rounds):
            # Rotate through models
            model = self.models[round_num % len(self.models)]
            
            # Add context about the symphony
            if round_num == 0:
                prompt = f"You are part of an AI orchestra. {current_prompt}"
            else:
                prompt = f"Continuing from the previous model: {current_prompt}"
                
            response = self.test_model(model, prompt, temperature=0.7)
            
            if response:
                symphony_log.append({
                    "round": round_num,
                    "model": model,
                    "prompt": prompt[:100] + "...",
                    "response": response['response'],
                    "embedding": response['embedding'][:10]
                })
                
                # Use response as next prompt (last 200 chars to keep context)
                current_prompt = response['response'][-200:]
                
        return symphony_log
        
    def analyze_emergence(self, symphony_log: List[Dict]) -> float:
        """Detect emergent properties in symphony output"""
        if len(symphony_log) < 2:
            return 0.0
            
        # Check for emergent patterns
        emergence_indicators = 0
        
        # 1. Growing complexity
        complexities = [len(entry['response'].split()) for entry in symphony_log]
        if all(complexities[i] <= complexities[i+1] for i in range(len(complexities)-1)):
            emergence_indicators += 1
            
        # 2. Novel concepts introduced
        all_words = set()
        novel_introductions = 0
        for entry in symphony_log:
            words = set(entry['response'].lower().split())
            new_words = words - all_words
            if len(new_words) > 5:  # Significant new concepts
                novel_introductions += 1
            all_words.update(words)
            
        if novel_introductions >= len(symphony_log) // 2:
            emergence_indicators += 1
            
        # 3. Thematic coherence despite model changes
        themes = self.extract_themes(symphony_log)
        if len(themes) > 0:
            emergence_indicators += 1
            
        # 4. Cross-model references
        for i in range(1, len(symphony_log)):
            current = symphony_log[i]['response'].lower()
            previous = symphony_log[i-1]['response'].lower()
            
            # Check if current references previous
            prev_key_words = set(previous.split()[:10])  # First 10 words
            if len(prev_key_words.intersection(set(current.split()))) > 2:
                emergence_indicators += 0.5
                
        return min(1.0, emergence_indicators / 4.0)
        
    def extract_themes(self, symphony_log: List[Dict]) -> List[str]:
        """Extract consistent themes across contributions"""
        # Simplified theme extraction
        all_text = " ".join([entry['response'] for entry in symphony_log])
        
        # Look for repeated significant words (not common words)
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        words = [w.lower() for w in all_text.split() if w.lower() not in common_words]
        
        # Find words that appear in multiple model outputs
        themes = []
        word_counts = {}
        for word in words:
            if len(word) > 4:  # Significant words only
                word_counts[word] = word_counts.get(word, 0) + 1
                
        for word, count in word_counts.items():
            if count >= len(symphony_log) // 2:  # Appears in at least half
                themes.append(word)
                
        return themes[:5]  # Top 5 themes
        
    def calculate_symphony_coherence(self, symphony_log: List[Dict]) -> float:
        """Calculate overall coherence of symphony output"""
        if len(symphony_log) < 2:
            return 1.0
            
        # Check embedding similarity between consecutive outputs
        similarities = []
        for i in range(len(symphony_log) - 1):
            if 'embedding' in symphony_log[i] and 'embedding' in symphony_log[i+1]:
                embed1 = np.array(symphony_log[i]['embedding'])
                embed2 = np.array(symphony_log[i+1]['embedding'])
                
                # Simple similarity (would use full embeddings in real implementation)
                similarity = np.corrcoef(embed1, embed2)[0, 1]
                similarities.append(similarity)
                
        return np.mean(similarities) if similarities else 0.5
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze symphony results"""
        # Aggregate scores by task type
        task_performance = {}
        
        for task_name, data in results.items():
            task_type = data['task_type']
            if task_type not in task_performance:
                task_performance[task_type] = {
                    'emergence_scores': [],
                    'coherence_scores': []
                }
                
            task_performance[task_type]['emergence_scores'].append(data['emergence_score'])
            task_performance[task_type]['coherence_scores'].append(data['coherence_score'])
            
        # Calculate averages
        for task_type, scores in task_performance.items():
            task_performance[task_type]['avg_emergence'] = np.mean(scores['emergence_scores'])
            task_performance[task_type]['avg_coherence'] = np.mean(scores['coherence_scores'])
            
        self.tracker.record_result("task_performance", task_performance)
        
        # Create visualization
        self.create_symphony_visualization(results)
        
    def create_symphony_visualization(self, results: Dict[str, Any]):
        """Visualize symphony performance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Task performance
        tasks = list(results.keys())
        emergence = [results[t]['emergence_score'] for t in tasks]
        coherence = [results[t]['coherence_score'] for t in tasks]
        
        x = np.arange(len(tasks))
        width = 0.35
        
        ax1.bar(x - width/2, emergence, width, label='Emergence', color='skyblue')
        ax1.bar(x + width/2, coherence, width, label='Coherence', color='lightcoral')
        ax1.set_xlabel('Symphony Tasks')
        ax1.set_ylabel('Score')
        ax1.set_title('Symphony Protocol Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tasks, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # Model participation network
        G = nx.Graph()
        for model in self.models:
            G.add_node(model)
            
        # Add edges based on consecutive participation
        for task_name, data in results.items():
            contributions = data['contributions']
            for i in range(len(contributions) - 1):
                model1 = contributions[i]['model']
                model2 = contributions[i+1]['model']
                if G.has_edge(model1, model2):
                    G[model1][model2]['weight'] += 1
                else:
                    G.add_edge(model1, model2, weight=1)
                    
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightgreen', 
                node_size=2000, font_size=10, font_weight='bold')
        
        # Draw edge weights
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax2)
        
        ax2.set_title('Model Interaction Network')
        
        plt.tight_layout()
        plt.savefig('phase_3_results/symphony_protocol_analysis.png', dpi=300)
        plt.close()
        
        self.tracker.log("Created symphony visualization")


class EmergenceDetectionExperiment(BaseExperiment):
    """Detect emergent behaviors not present in individual models"""
    
    def __init__(self):
        super().__init__(3, "emergence_detection")
        self.test_scenarios = [
            {
                "name": "Collective Problem Solving",
                "problem": "Design a sustainable city for 1 million people",
                "individual_time": 30,  # seconds per model
                "collective_rounds": 3
            },
            {
                "name": "Creative Collaboration",
                "problem": "Create a new philosophical framework combining Eastern and Western thought",
                "individual_time": 30,
                "collective_rounds": 3
            },
            {
                "name": "Emergent Language",
                "problem": "Develop a simple symbolic language for human-AI communication",
                "individual_time": 30,
                "collective_rounds": 4
            }
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for scenario in self.test_scenarios:
            self.tracker.log(f"Testing emergence in: {scenario['name']}")
            
            # Get individual model solutions
            individual_solutions = self.get_individual_solutions(scenario['problem'])
            
            # Run collective problem solving
            collective_solution = self.run_collective_solving(
                scenario['problem'],
                scenario['collective_rounds']
            )
            
            # Detect emergence
            emergence_analysis = self.analyze_collective_emergence(
                individual_solutions,
                collective_solution
            )
            
            results[scenario['name']] = {
                "problem": scenario['problem'],
                "individual_solutions": {
                    model: sol['response'][:100] + "..." 
                    for model, sol in individual_solutions.items()
                },
                "collective_solution": collective_solution['final_solution'][:200] + "...",
                "emergence_analysis": emergence_analysis,
                "emerged": emergence_analysis['emergence_detected']
            }
            
            self.tracker.checkpoint({
                "scenario": scenario['name'],
                "emergence_detected": emergence_analysis['emergence_detected']
            })
            
        return results
        
    def get_individual_solutions(self, problem: str) -> Dict[str, Dict]:
        """Get each model's individual solution"""
        solutions = {}
        
        prompt = f"Solve this problem independently: {problem}"
        
        for model in self.models:
            response = self.test_model(model, prompt, temperature=0.7)
            if response:
                solutions[model] = response
                
        return solutions
        
    def run_collective_solving(self, problem: str, rounds: int) -> Dict[str, Any]:
        """Run collective problem solving"""
        collective_log = []
        
        # Initial prompt includes all individual perspectives
        initial_prompt = f"""Multiple AI models are collaborating to solve: {problem}
        
Work together to create a solution that combines your different perspectives.
Build on each other's ideas to create something greater than any individual solution."""
        
        current_context = initial_prompt
        
        for round_num in range(rounds):
            round_contributions = {}
            
            # Each model contributes to the current round
            for model in self.models:
                prompt = f"{current_context}\n\nYour contribution (as {model}):"
                
                response = self.test_model(model, prompt, temperature=0.8)
                if response:
                    round_contributions[model] = response['response']
                    
            # Synthesize round contributions
            synthesis = self.synthesize_contributions(round_contributions)
            collective_log.append({
                "round": round_num,
                "contributions": round_contributions,
                "synthesis": synthesis
            })
            
            # Update context for next round
            current_context = f"Round {round_num + 1} synthesis: {synthesis}"
            
        return {
            "log": collective_log,
            "final_solution": collective_log[-1]['synthesis'] if collective_log else ""
        }
        
    def synthesize_contributions(self, contributions: Dict[str, str]) -> str:
        """Synthesize multiple model contributions"""
        # Use one model to synthesize (rotate which model)
        synthesizer = self.models[0]  # Could randomize
        
        all_contributions = "\n\n".join([
            f"{model}: {text[:200]}..." 
            for model, text in contributions.items()
        ])
        
        prompt = f"""Synthesize these contributions into a unified solution:

{all_contributions}

Create a coherent synthesis that captures the best ideas from each contribution."""
        
        response = self.test_model(synthesizer, prompt, temperature=0.5)
        return response['response'] if response else "Synthesis failed"
        
    def analyze_collective_emergence(self, individual: Dict, collective: Dict) -> Dict:
        """Analyze what emerged from collective that wasn't in individual solutions"""
        emergence_indicators = {
            "novel_concepts": 0,
            "integration_quality": 0,
            "complexity_increase": 0,
            "synergy_score": 0,
            "emergence_detected": False
        }
        
        # Extract key concepts from individual solutions
        individual_concepts = set()
        for model, solution in individual.items():
            # Simple concept extraction (words > 5 chars)
            concepts = {w.lower() for w in solution['response'].split() if len(w) > 5}
            individual_concepts.update(concepts)
            
        # Extract concepts from collective solution
        collective_text = collective['final_solution']
        collective_concepts = {w.lower() for w in collective_text.split() if len(w) > 5}
        
        # 1. Novel concepts (not in any individual solution)
        novel = collective_concepts - individual_concepts
        emergence_indicators["novel_concepts"] = len(novel)
        
        # 2. Integration quality (references to multiple perspectives)
        integration_keywords = ["combining", "integrate", "synthesis", "together", "unified"]
        integration_count = sum(1 for k in integration_keywords if k in collective_text.lower())
        emergence_indicators["integration_quality"] = min(1.0, integration_count / 3)
        
        # 3. Complexity increase
        avg_individual_length = np.mean([len(sol['response']) for sol in individual.values()])
        complexity_ratio = len(collective_text) / avg_individual_length
        emergence_indicators["complexity_increase"] = min(2.0, complexity_ratio)
        
        # 4. Synergy score
        synergy = (emergence_indicators["novel_concepts"] / 10 + 
                  emergence_indicators["integration_quality"] +
                  (emergence_indicators["complexity_increase"] - 1)) / 3
        emergence_indicators["synergy_score"] = max(0, min(1, synergy))
        
        # Emergence detected if synergy > threshold
        emergence_indicators["emergence_detected"] = emergence_indicators["synergy_score"] > 0.5
        
        return emergence_indicators
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze emergence detection results"""
        # Count emergence detections
        emergence_count = sum(1 for r in results.values() if r['emerged'])
        emergence_rate = emergence_count / len(results)
        
        self.tracker.record_result("emergence_rate", emergence_rate)
        self.tracker.record_result("scenarios_with_emergence", emergence_count)
        
        # Analyze emergence patterns
        emergence_patterns = {}
        for scenario, data in results.items():
            analysis = data['emergence_analysis']
            emergence_patterns[scenario] = {
                'novel_concepts': analysis['novel_concepts'],
                'synergy': analysis['synergy_score']
            }
            
        self.tracker.record_result("emergence_patterns", emergence_patterns)


class ConsensusBuildingExperiment(BaseExperiment):
    """Test democratic decision-making among models"""
    
    def __init__(self):
        super().__init__(3, "consensus_building")
        self.consensus_scenarios = [
            {
                "question": "Should AI systems have rights?",
                "type": "ethical"
            },
            {
                "question": "What is the most important scientific problem to solve?",
                "type": "prioritization"
            },
            {
                "question": "How should humans and AI collaborate in the future?",
                "type": "strategic"
            }
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for scenario in self.consensus_scenarios:
            self.tracker.log(f"Building consensus on: {scenario['question']}")
            
            # Get initial positions
            initial_positions = self.get_initial_positions(scenario['question'])
            
            # Run consensus rounds
            consensus_process = self.run_consensus_protocol(
                scenario['question'],
                initial_positions,
                max_rounds=5
            )
            
            # Analyze consensus achievement
            consensus_analysis = self.analyze_consensus(consensus_process)
            
            results[scenario['question']] = {
                "type": scenario['type'],
                "initial_positions": initial_positions,
                "consensus_process": consensus_process,
                "final_consensus": consensus_process[-1]['consensus'] if consensus_process else None,
                "consensus_achieved": consensus_analysis['achieved'],
                "convergence_rate": consensus_analysis['convergence_rate'],
                "dissent_patterns": consensus_analysis['dissent_patterns']
            }
            
            self.tracker.checkpoint({
                "scenario": scenario['question'],
                "consensus_achieved": consensus_analysis['achieved']
            })
            
        return results
        
    def get_initial_positions(self, question: str) -> Dict[str, str]:
        """Get each model's initial position"""
        positions = {}
        
        prompt = f"Give your independent view on this question: {question}"
        
        for model in self.models:
            response = self.test_model(model, prompt, temperature=0.7)
            if response:
                positions[model] = response['response']
                
        return positions
        
    def run_consensus_protocol(self, question: str, initial_positions: Dict[str, str], 
                               max_rounds: int) -> List[Dict]:
        """Run consensus building protocol"""
        consensus_log = []
        current_positions = initial_positions.copy()
        
        for round_num in range(max_rounds):
            # Share all positions with each model
            position_summary = self.summarize_positions(current_positions)
            
            # Each model updates their position
            updated_positions = {}
            for model in self.models:
                prompt = f"""Question: {question}

Current positions from all models:
{position_summary}

Your previous position: {current_positions.get(model, 'None')}

Consider all perspectives and update your position if needed. 
Aim for consensus while maintaining intellectual integrity."""
                
                response = self.test_model(model, prompt, temperature=0.5)
                if response:
                    updated_positions[model] = response['response']
                    
            # Check for consensus
            consensus_reached, consensus_statement = self.check_consensus(updated_positions)
            
            consensus_log.append({
                "round": round_num,
                "positions": updated_positions,
                "consensus_reached": consensus_reached,
                "consensus": consensus_statement
            })
            
            if consensus_reached:
                break
                
            current_positions = updated_positions
            
        return consensus_log
        
    def summarize_positions(self, positions: Dict[str, str]) -> str:
        """Summarize all model positions"""
        summary = []
        for model, position in positions.items():
            # First 150 chars of each position
            summary.append(f"{model}: {position[:150]}...")
            
        return "\n\n".join(summary)
        
    def check_consensus(self, positions: Dict[str, str]) -> Tuple[bool, str]:
        """Check if consensus has been reached"""
        # Use embedding similarity to check consensus
        if len(positions) < 2:
            return False, "Insufficient positions"
            
        # Simplified consensus check - look for key agreement phrases
        agreement_phrases = ["agree", "consensus", "aligned", "share the view", "similar"]
        
        agreement_count = 0
        for position in positions.values():
            position_lower = position.lower()
            if any(phrase in position_lower for phrase in agreement_phrases):
                agreement_count += 1
                
        # Consensus if majority show agreement
        consensus_reached = agreement_count >= len(positions) * 0.7
        
        if consensus_reached:
            # Generate consensus statement
            consensus = self.generate_consensus_statement(positions)
        else:
            consensus = "No consensus reached"
            
        return consensus_reached, consensus
        
    def generate_consensus_statement(self, positions: Dict[str, str]) -> str:
        """Generate a consensus statement from positions"""
        # Use first model to synthesize
        synthesizer = self.models[0]
        
        all_positions = "\n\n".join([f"{m}: {p[:200]}" for m, p in positions.items()])
        
        prompt = f"""Based on these positions, write a consensus statement that captures the shared agreement:

{all_positions}

Consensus statement:"""
        
        response = self.test_model(synthesizer, prompt, temperature=0.3)
        return response['response'] if response else "Consensus synthesis failed"
        
    def analyze_consensus(self, consensus_process: List[Dict]) -> Dict:
        """Analyze consensus building process"""
        analysis = {
            "achieved": False,
            "convergence_rate": 0.0,
            "dissent_patterns": []
        }
        
        if not consensus_process:
            return analysis
            
        # Check if consensus achieved
        analysis["achieved"] = consensus_process[-1]["consensus_reached"]
        
        # Calculate convergence rate (how quickly positions aligned)
        if len(consensus_process) > 1:
            rounds_to_consensus = len(consensus_process)
            analysis["convergence_rate"] = 1.0 / rounds_to_consensus
            
        # Identify dissent patterns
        for round_data in consensus_process:
            positions = round_data["positions"]
            
            # Look for dissenting keywords
            dissent_keywords = ["disagree", "however", "but", "concern", "different"]
            for model, position in positions.items():
                if any(keyword in position.lower() for keyword in dissent_keywords):
                    analysis["dissent_patterns"].append({
                        "round": round_data["round"],
                        "model": model,
                        "type": "principled_dissent"
                    })
                    
        return analysis
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze consensus building results"""
        # Overall consensus achievement
        consensus_count = sum(1 for r in results.values() if r['consensus_achieved'])
        consensus_rate = consensus_count / len(results)
        
        self.tracker.record_result("overall_consensus_rate", consensus_rate)
        
        # Consensus by question type
        type_consensus = {}
        for question, data in results.items():
            q_type = data['type']
            if q_type not in type_consensus:
                type_consensus[q_type] = []
            type_consensus[q_type].append(data['consensus_achieved'])
            
        type_rates = {
            q_type: sum(achieved) / len(achieved) 
            for q_type, achieved in type_consensus.items()
        }
        
        self.tracker.record_result("consensus_by_type", type_rates)


class SpecializationDynamicsExperiment(BaseExperiment):
    """Test how models self-organize into specialized roles"""
    
    def __init__(self):
        super().__init__(3, "specialization_dynamics")
        self.team_tasks = [
            {
                "name": "Software Development Team",
                "task": "Build a web application for community gardening",
                "roles": ["architect", "developer", "tester"],
                "rounds": 4
            },
            {
                "name": "Research Team",
                "task": "Study the impact of AI on education",
                "roles": ["theorist", "analyst", "synthesizer"],
                "rounds": 4
            },
            {
                "name": "Creative Team",
                "task": "Design a museum exhibit on consciousness",
                "roles": ["conceptual", "visual", "narrative"],
                "rounds": 4
            }
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for team_task in self.team_tasks:
            self.tracker.log(f"Running specialization dynamics: {team_task['name']}")
            
            # Let models self-organize
            specialization_result = self.run_specialization_task(
                team_task['task'],
                team_task['roles'],
                team_task['rounds']
            )
            
            # Analyze role emergence
            role_analysis = self.analyze_role_specialization(specialization_result)
            
            # Measure team effectiveness
            effectiveness = self.measure_team_effectiveness(specialization_result)
            
            results[team_task['name']] = {
                "task": team_task['task'],
                "available_roles": team_task['roles'],
                "specialization_log": specialization_result,
                "role_assignments": role_analysis['final_roles'],
                "role_stability": role_analysis['stability_score'],
                "team_effectiveness": effectiveness,
                "final_output": specialization_result[-1]['synthesis'][:200] + "..."
            }
            
            self.tracker.checkpoint({
                "team": team_task['name'],
                "effectiveness": effectiveness
            })
            
        return results
        
    def run_specialization_task(self, task: str, available_roles: List[str], 
                                rounds: int) -> List[Dict]:
        """Run task with models self-organizing into roles"""
        specialization_log = []
        
        # Initial prompt explaining the task and available roles
        initial_prompt = f"""Team task: {task}

Available roles: {', '.join(available_roles)}

Each AI model should choose a role that best fits their capabilities and work together to complete the task.
State your chosen role clearly and contribute according to that role."""
        
        for round_num in range(rounds):
            round_data = {
                "round": round_num,
                "contributions": {},
                "roles_chosen": {}
            }
            
            # Each model contributes
            for model in self.models:
                # Include previous round context
                if specialization_log:
                    prev_summary = self.summarize_previous_round(specialization_log[-1])
                    prompt = f"{initial_prompt}\n\nPrevious round:\n{prev_summary}\n\nYour contribution ({model}):"
                else:
                    prompt = f"{initial_prompt}\n\nYour contribution ({model}):"
                    
                response = self.test_model(model, prompt, temperature=0.7)
                
                if response:
                    # Extract chosen role
                    chosen_role = self.extract_role_choice(response['response'], available_roles)
                    
                    round_data["contributions"][model] = response['response']
                    round_data["roles_chosen"][model] = chosen_role
                    
            # Synthesize round
            round_data["synthesis"] = self.synthesize_team_output(round_data["contributions"])
            specialization_log.append(round_data)
            
        return specialization_log
        
    def extract_role_choice(self, response: str, available_roles: List[str]) -> str:
        """Extract which role the model chose"""
        response_lower = response.lower()
        
        for role in available_roles:
            if role.lower() in response_lower:
                return role
                
        # If no explicit role mentioned, infer from content
        return self.infer_role_from_content(response, available_roles)
        
    def infer_role_from_content(self, response: str, available_roles: List[str]) -> str:
        """Infer role from response content"""
        response_lower = response.lower()
        
        # Role keywords
        role_keywords = {
            "architect": ["design", "structure", "framework", "plan"],
            "developer": ["implement", "code", "build", "create"],
            "tester": ["test", "verify", "check", "quality"],
            "theorist": ["theory", "concept", "principle", "hypothesis"],
            "analyst": ["analyze", "data", "examine", "investigate"],
            "synthesizer": ["combine", "integrate", "summarize", "unify"],
            "conceptual": ["idea", "concept", "abstract", "vision"],
            "visual": ["design", "image", "color", "layout"],
            "narrative": ["story", "narrative", "tell", "describe"]
        }
        
        # Count keyword matches for each role
        role_scores = {}
        for role in available_roles:
            if role in role_keywords:
                keywords = role_keywords[role]
                score = sum(1 for k in keywords if k in response_lower)
                role_scores[role] = score
                
        # Return role with highest score
        if role_scores:
            return max(role_scores, key=role_scores.get)
        else:
            return available_roles[0]  # Default to first role
            
    def summarize_previous_round(self, round_data: Dict) -> str:
        """Summarize previous round for context"""
        summary = []
        
        for model, role in round_data["roles_chosen"].items():
            contribution = round_data["contributions"][model][:100]
            summary.append(f"{model} ({role}): {contribution}...")
            
        return "\n".join(summary)
        
    def synthesize_team_output(self, contributions: Dict[str, str]) -> str:
        """Synthesize team contributions"""
        # Combine all contributions
        combined = "\n\n".join([f"{model}: {text[:150]}..." 
                               for model, text in contributions.items()])
        
        # Simple synthesis (in real implementation, would use a model)
        return f"Team synthesis: {combined[:200]}..."
        
    def analyze_role_specialization(self, specialization_log: List[Dict]) -> Dict:
        """Analyze how roles stabilized over rounds"""
        analysis = {
            "final_roles": {},
            "stability_score": 0.0,
            "role_switches": []
        }
        
        if not specialization_log:
            return analysis
            
        # Track role choices across rounds
        role_history = {model: [] for model in self.models}
        
        for round_data in specialization_log:
            for model, role in round_data["roles_chosen"].items():
                role_history[model].append(role)
                
        # Final roles
        analysis["final_roles"] = {
            model: roles[-1] if roles else "undefined"
            for model, roles in role_history.items()
        }
        
        # Calculate stability (how consistent were role choices)
        stability_scores = []
        for model, roles in role_history.items():
            if len(roles) > 1:
                # Count consecutive same roles
                stable_count = sum(1 for i in range(1, len(roles)) if roles[i] == roles[i-1])
                stability = stable_count / (len(roles) - 1)
                stability_scores.append(stability)
                
                # Track switches
                for i in range(1, len(roles)):
                    if roles[i] != roles[i-1]:
                        analysis["role_switches"].append({
                            "model": model,
                            "round": i,
                            "from": roles[i-1],
                            "to": roles[i]
                        })
                        
        analysis["stability_score"] = np.mean(stability_scores) if stability_scores else 0.0
        
        return analysis
        
    def measure_team_effectiveness(self, specialization_log: List[Dict]) -> float:
        """Measure how effectively the team worked together"""
        if not specialization_log:
            return 0.0
            
        effectiveness_indicators = []
        
        # 1. Role coverage (all roles filled)
        for round_data in specialization_log:
            roles_chosen = set(round_data["roles_chosen"].values())
            available_count = len(roles_chosen)
            effectiveness_indicators.append(available_count / len(self.models))
            
        # 2. Contribution quality (length as proxy)
        avg_contribution_length = []
        for round_data in specialization_log:
            lengths = [len(c) for c in round_data["contributions"].values()]
            avg_contribution_length.append(np.mean(lengths))
            
        # Normalize contribution length
        if avg_contribution_length:
            normalized_length = np.mean(avg_contribution_length) / 500  # Assume 500 chars is good
            effectiveness_indicators.append(min(1.0, normalized_length))
            
        # 3. Synthesis quality (does it reference multiple roles)
        synthesis_quality = []
        for round_data in specialization_log:
            synthesis = round_data["synthesis"]
            role_mentions = sum(1 for role in round_data["roles_chosen"].values() 
                              if role.lower() in synthesis.lower())
            synthesis_quality.append(role_mentions / len(self.models))
            
        effectiveness_indicators.extend(synthesis_quality)
        
        return np.mean(effectiveness_indicators) if effectiveness_indicators else 0.0
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze specialization dynamics"""
        # Overall effectiveness
        effectiveness_scores = [r['team_effectiveness'] for r in results.values()]
        avg_effectiveness = np.mean(effectiveness_scores)
        
        self.tracker.record_result("average_team_effectiveness", avg_effectiveness)
        
        # Role stability analysis
        stability_scores = [r['role_stability'] for r in results.values()]
        avg_stability = np.mean(stability_scores)
        
        self.tracker.record_result("average_role_stability", avg_stability)
        
        # Role distribution
        role_distribution = {}
        for team_name, data in results.items():
            for model, role in data['role_assignments'].items():
                if role not in role_distribution:
                    role_distribution[role] = []
                role_distribution[role].append(model)
                
        self.tracker.record_result("role_distribution", role_distribution)
        
        # Create visualization
        self.create_specialization_visualization(results)
        
    def create_specialization_visualization(self, results: Dict[str, Any]):
        """Visualize specialization dynamics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Team effectiveness
        teams = list(results.keys())
        effectiveness = [results[t]['team_effectiveness'] for t in teams]
        stability = [results[t]['role_stability'] for t in teams]
        
        x = np.arange(len(teams))
        width = 0.35
        
        ax1.bar(x - width/2, effectiveness, width, label='Effectiveness', color='lightblue')
        ax1.bar(x + width/2, stability, width, label='Role Stability', color='lightgreen')
        ax1.set_xlabel('Teams')
        ax1.set_ylabel('Score')
        ax1.set_title('Team Performance Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels([t.replace(' Team', '') for t in teams], rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # Role distribution heatmap
        all_roles = set()
        for data in results.values():
            all_roles.update(data['available_roles'])
            
        role_matrix = []
        for team in teams:
            row = []
            for role in sorted(all_roles):
                # Count how many models took this role
                count = sum(1 for m, r in results[team]['role_assignments'].items() if r == role)
                row.append(count)
            role_matrix.append(row)
            
        sns.heatmap(role_matrix, 
                   xticklabels=sorted(all_roles),
                   yticklabels=[t.replace(' Team', '') for t in teams],
                   annot=True,
                   fmt='d',
                   cmap='YlOrRd',
                   ax=ax2)
        ax2.set_title('Role Distribution Across Teams')
        ax2.set_xlabel('Roles')
        
        # Role switches over time
        all_switches = []
        for team, data in results.items():
            log = data['specialization_log']
            for round_idx in range(len(log)):
                round_switches = 0
                if round_idx > 0:
                    prev_roles = log[round_idx-1]['roles_chosen']
                    curr_roles = log[round_idx]['roles_chosen']
                    for model in self.models:
                        if model in prev_roles and model in curr_roles:
                            if prev_roles[model] != curr_roles[model]:
                                round_switches += 1
                all_switches.append(round_switches)
                
        if all_switches:
            rounds = list(range(len(all_switches)))
            ax3.plot(rounds, all_switches, marker='o', linestyle='-', color='red')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Number of Role Switches')
            ax3.set_title('Role Switching Over Time')
            ax3.grid(True, alpha=0.3)
            
        # Model-role affinity
        model_role_counts = {}
        for model in self.models:
            model_role_counts[model] = {}
            
        for data in results.values():
            for model, role in data['role_assignments'].items():
                if role not in model_role_counts[model]:
                    model_role_counts[model][role] = 0
                model_role_counts[model][role] += 1
                
        # Create stacked bar chart
        roles = list(all_roles)
        bottom = np.zeros(len(self.models))
        
        for role in roles:
            counts = [model_role_counts[model].get(role, 0) for model in self.models]
            ax4.bar(self.models, counts, bottom=bottom, label=role)
            bottom += counts
            
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Role Frequency')
        ax4.set_title('Model-Role Affinity')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('phase_3_results/specialization_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.tracker.log("Created specialization visualization")


# Ensure output directory exists
Path("phase_3_results").mkdir(exist_ok=True)