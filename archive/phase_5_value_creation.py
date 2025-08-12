#!/usr/bin/env python3
"""
Phase 5: Value Creation Chains Experiments
Exploring how AI systems create cascading value through collaborative chains
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
import time

from experiment_tracker import ExperimentTracker
from autonomous_experiment_runner import BaseExperiment


class ValuePropagationTestExperiment(BaseExperiment):
    """Test how value flows through model chains"""
    
    def __init__(self):
        super().__init__(5, "value_propagation_test")
        self.value_seeds = [
            {
                "type": "knowledge",
                "seed": "E=mc²",
                "description": "Scientific knowledge seed"
            },
            {
                "type": "creative",
                "seed": "A story begins with a single word",
                "description": "Creative inspiration seed"
            },
            {
                "type": "solution",
                "seed": "Reduce, reuse, recycle",
                "description": "Problem-solving seed"
            },
            {
                "type": "philosophical",
                "seed": "What is the meaning of existence?",
                "description": "Philosophical inquiry seed"
            }
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for seed_data in self.value_seeds:
            self.tracker.log(f"Testing value propagation for: {seed_data['type']}")
            
            # Create value chain
            chain_length = 5
            propagation_chain = self.propagate_value(
                seed_data['seed'],
                chain_length,
                seed_data['type']
            )
            
            # Analyze value growth
            value_analysis = self.analyze_value_growth(propagation_chain)
            
            # Measure value quality
            quality_metrics = self.measure_value_quality(propagation_chain)
            
            results[seed_data['type']] = {
                "seed": seed_data['seed'],
                "chain": propagation_chain,
                "value_analysis": value_analysis,
                "quality_metrics": quality_metrics,
                "total_value_created": value_analysis['total_value']
            }
            
            self.tracker.checkpoint({
                "seed_type": seed_data['type'],
                "total_value": value_analysis['total_value']
            })
            
        return results
        
    def propagate_value(self, seed: str, chain_length: int, value_type: str) -> List[Dict]:
        """Propagate value through a chain of models"""
        chain = []
        current_value = seed
        
        for step in range(chain_length):
            # Rotate through models
            model = self.models[step % len(self.models)]
            
            # Create prompt based on value type
            if step == 0:
                prompt = f"Given this seed value: '{seed}'\nExpand, enhance, or build upon it to create more value:"
            else:
                prompt = f"Previous value: {current_value}\n\nBuild upon this to create even more value:"
                
            response = self.test_model(model, prompt, temperature=0.7)
            
            if response:
                # Extract value metrics
                value_metrics = self.extract_value_metrics(
                    response['response'],
                    current_value,
                    value_type
                )
                
                chain.append({
                    "step": step,
                    "model": model,
                    "input": current_value[:100] + "...",
                    "output": response['response'],
                    "value_metrics": value_metrics,
                    "value_added": value_metrics['value_score']
                })
                
                # Update current value for next iteration
                current_value = response['response']
                
        return chain
        
    def extract_value_metrics(self, output: str, input_text: str, value_type: str) -> Dict:
        """Extract metrics about value creation"""
        metrics = {}
        
        # Novelty: How much new content was added
        input_words = set(input_text.lower().split())
        output_words = set(output.lower().split())
        new_words = output_words - input_words
        
        metrics['novelty'] = len(new_words) / (len(output_words) + 1)
        
        # Relevance: How well it relates to input
        common_words = input_words.intersection(output_words)
        metrics['relevance'] = len(common_words) / (len(input_words) + 1)
        
        # Complexity: Vocabulary diversity
        metrics['complexity'] = len(set(output.split())) / (len(output.split()) + 1)
        
        # Type-specific value
        type_keywords = {
            "knowledge": ["understand", "learn", "discover", "explain", "theory"],
            "creative": ["imagine", "create", "story", "beautiful", "unique"],
            "solution": ["solve", "improve", "efficient", "better", "optimize"],
            "philosophical": ["meaning", "purpose", "why", "truth", "existence"]
        }
        
        keywords = type_keywords.get(value_type, [])
        keyword_count = sum(1 for k in keywords if k in output.lower())
        metrics['type_alignment'] = keyword_count / (len(keywords) + 1)
        
        # Overall value score
        metrics['value_score'] = (
            metrics['novelty'] * 0.3 +
            metrics['relevance'] * 0.2 +
            metrics['complexity'] * 0.2 +
            metrics['type_alignment'] * 0.3
        )
        
        return metrics
        
    def analyze_value_growth(self, chain: List[Dict]) -> Dict:
        """Analyze how value grows through the chain"""
        if not chain:
            return {"total_value": 0, "growth_pattern": "none"}
            
        # Extract value scores
        value_scores = [step['value_added'] for step in chain]
        
        # Calculate cumulative value
        cumulative_value = np.cumsum(value_scores)
        
        # Determine growth pattern
        if len(value_scores) > 1:
            # Check if values are increasing
            differences = np.diff(value_scores)
            if all(d > 0 for d in differences):
                growth_pattern = "exponential"
            elif all(d >= 0 for d in differences):
                growth_pattern = "linear"
            elif sum(differences) > 0:
                growth_pattern = "variable"
            else:
                growth_pattern = "diminishing"
        else:
            growth_pattern = "insufficient_data"
            
        return {
            "total_value": float(cumulative_value[-1]) if len(cumulative_value) > 0 else 0,
            "growth_pattern": growth_pattern,
            "value_trajectory": value_scores,
            "cumulative_trajectory": cumulative_value.tolist() if len(cumulative_value) > 0 else [],
            "average_step_value": np.mean(value_scores) if value_scores else 0
        }
        
    def measure_value_quality(self, chain: List[Dict]) -> Dict:
        """Measure quality aspects of created value"""
        if not chain:
            return {}
            
        # Coherence: How well each step follows from previous
        coherence_scores = []
        for i in range(1, len(chain)):
            prev_output = chain[i-1]['output']
            curr_output = chain[i]['output']
            
            # Simple coherence based on shared concepts
            prev_concepts = set(prev_output.lower().split())
            curr_concepts = set(curr_output.lower().split())
            
            overlap = len(prev_concepts.intersection(curr_concepts))
            coherence = overlap / (min(len(prev_concepts), len(curr_concepts)) + 1)
            coherence_scores.append(coherence)
            
        # Diversity: How different are the contributions
        all_outputs = [step['output'] for step in chain]
        diversity = self.calculate_diversity(all_outputs)
        
        # Depth: Average complexity across chain
        complexities = [step['value_metrics']['complexity'] for step in chain]
        
        return {
            "coherence": np.mean(coherence_scores) if coherence_scores else 0,
            "diversity": diversity,
            "depth": np.mean(complexities) if complexities else 0,
            "consistency": np.std(complexities) if len(complexities) > 1 else 0
        }
        
    def calculate_diversity(self, texts: List[str]) -> float:
        """Calculate diversity of text contributions"""
        if len(texts) < 2:
            return 0
            
        # Use vocabulary diversity as proxy
        all_vocabs = []
        for text in texts:
            vocab = set(text.lower().split())
            all_vocabs.append(vocab)
            
        # Calculate pairwise differences
        diversity_scores = []
        for i in range(len(all_vocabs)):
            for j in range(i+1, len(all_vocabs)):
                unique_to_i = len(all_vocabs[i] - all_vocabs[j])
                unique_to_j = len(all_vocabs[j] - all_vocabs[i])
                total = len(all_vocabs[i]) + len(all_vocabs[j])
                
                diversity = (unique_to_i + unique_to_j) / (total + 1)
                diversity_scores.append(diversity)
                
        return np.mean(diversity_scores) if diversity_scores else 0
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze value propagation results"""
        # Compare value creation across types
        value_summary = {}
        
        for value_type, data in results.items():
            value_summary[value_type] = {
                "total_value": data['total_value_created'],
                "growth_pattern": data['value_analysis']['growth_pattern'],
                "quality_score": np.mean([
                    data['quality_metrics']['coherence'],
                    data['quality_metrics']['diversity'],
                    data['quality_metrics']['depth']
                ])
            }
            
        self.tracker.record_result("value_summary", value_summary)
        
        # Find best value creation type
        best_type = max(value_summary.items(), 
                       key=lambda x: x[1]['total_value'])
        self.tracker.record_result("best_value_type", best_type)
        
        # Create visualization
        self.create_value_propagation_visualization(results)
        
    def create_value_propagation_visualization(self, results: Dict[str, Any]):
        """Visualize value propagation"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Value growth trajectories
        for value_type, data in results.items():
            trajectory = data['value_analysis']['cumulative_trajectory']
            if trajectory:
                ax1.plot(range(len(trajectory)), trajectory, 
                        marker='o', label=value_type)
                
        ax1.set_xlabel('Chain Step')
        ax1.set_ylabel('Cumulative Value')
        ax1.set_title('Value Growth Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Value types comparison
        types = list(results.keys())
        total_values = [results[t]['total_value_created'] for t in types]
        
        ax2.bar(types, total_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        ax2.set_xlabel('Value Type')
        ax2.set_ylabel('Total Value Created')
        ax2.set_title('Value Creation by Type')
        ax2.tick_params(axis='x', rotation=45)
        
        # Quality metrics heatmap
        quality_data = []
        metrics = ['coherence', 'diversity', 'depth']
        
        for value_type in types:
            row = [results[value_type]['quality_metrics'][m] for m in metrics]
            quality_data.append(row)
            
        sns.heatmap(quality_data, 
                   xticklabels=metrics,
                   yticklabels=types,
                   annot=True,
                   fmt='.2f',
                   cmap='YlGn',
                   ax=ax3)
        ax3.set_title('Value Quality Metrics')
        
        # Growth patterns
        growth_patterns = {}
        for value_type, data in results.items():
            pattern = data['value_analysis']['growth_pattern']
            growth_patterns[pattern] = growth_patterns.get(pattern, 0) + 1
            
        if growth_patterns:
            ax4.pie(growth_patterns.values(), 
                   labels=growth_patterns.keys(),
                   autopct='%1.0f%%',
                   colors=['gold', 'lightblue', 'lightgreen', 'pink'])
            ax4.set_title('Growth Pattern Distribution')
        
        plt.tight_layout()
        plt.savefig('phase_5_results/value_propagation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.tracker.log("Created value propagation visualization")


class EmergentValueDiscoveryExperiment(BaseExperiment):
    """Discover unexpected value creation in AI collaborations"""
    
    def __init__(self):
        super().__init__(5, "emergent_value_discovery")
        self.collaboration_scenarios = [
            {
                "name": "Cross-Domain Fusion",
                "domains": ["poetry", "mathematics", "cooking"],
                "prompt": "Create something valuable by combining"
            },
            {
                "name": "Problem Synthesis",
                "domains": ["climate change", "education", "technology"],
                "prompt": "Find unexpected connections between"
            },
            {
                "name": "Abstract Concretization",
                "domains": ["love", "quantum physics", "democracy"],
                "prompt": "Make tangible connections between"
            }
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for scenario in self.collaboration_scenarios:
            self.tracker.log(f"Testing emergent value in: {scenario['name']}")
            
            # Individual domain exploration
            individual_values = self.explore_domains_individually(
                scenario['domains'],
                scenario['prompt']
            )
            
            # Collaborative fusion
            emergent_value = self.create_emergent_fusion(
                scenario['domains'],
                scenario['prompt'],
                individual_values
            )
            
            # Analyze emergence
            emergence_analysis = self.analyze_emergence(
                individual_values,
                emergent_value
            )
            
            results[scenario['name']] = {
                "domains": scenario['domains'],
                "individual_values": individual_values,
                "emergent_value": emergent_value,
                "emergence_analysis": emergence_analysis,
                "emergence_factor": emergence_analysis['emergence_factor']
            }
            
            self.tracker.checkpoint({
                "scenario": scenario['name'],
                "emergence_factor": emergence_analysis['emergence_factor']
            })
            
        return results
        
    def explore_domains_individually(self, domains: List[str], base_prompt: str) -> Dict:
        """Explore each domain individually"""
        individual_values = {}
        
        for i, domain in enumerate(domains):
            model = self.models[i % len(self.models)]
            
            prompt = f"{base_prompt} {domain}. Focus only on this domain:"
            
            response = self.test_model(model, prompt, temperature=0.7)
            
            if response:
                individual_values[domain] = {
                    "model": model,
                    "response": response['response'],
                    "value_metrics": self.assess_domain_value(response['response'], domain)
                }
                
        return individual_values
        
    def create_emergent_fusion(self, domains: List[str], base_prompt: str, 
                              individual_values: Dict) -> Dict:
        """Create emergent value through fusion"""
        # Prepare context from individual explorations
        context_parts = []
        for domain, data in individual_values.items():
            summary = data['response'][:150] + "..."
            context_parts.append(f"{domain}: {summary}")
            
        context = "\n\n".join(context_parts)
        
        # Multi-round fusion process
        fusion_rounds = []
        current_fusion = ""
        
        for round_num in range(3):  # 3 rounds of fusion
            model = self.models[round_num % len(self.models)]
            
            if round_num == 0:
                prompt = f"""Given these individual domain explorations:

{context}

Now {base_prompt} ALL of these domains together to create something entirely new and valuable that emerges from their intersection:"""
            else:
                prompt = f"""Previous fusion:
{current_fusion}

Deepen and expand this fusion, finding even more unexpected connections and emergent value:"""
                
            response = self.test_model(model, prompt, temperature=0.8)
            
            if response:
                fusion_rounds.append({
                    "round": round_num,
                    "model": model,
                    "response": response['response'],
                    "novelty": self.assess_novelty(response['response'], individual_values)
                })
                
                current_fusion = response['response']
                
        return {
            "fusion_rounds": fusion_rounds,
            "final_fusion": current_fusion,
            "total_novelty": np.mean([r['novelty'] for r in fusion_rounds])
        }
        
    def assess_domain_value(self, response: str, domain: str) -> Dict:
        """Assess value created for a single domain"""
        # Domain-specific keywords
        domain_keywords = {
            "poetry": ["verse", "rhythm", "metaphor", "beauty", "emotion"],
            "mathematics": ["equation", "proof", "pattern", "logic", "theorem"],
            "cooking": ["flavor", "ingredient", "recipe", "taste", "technique"],
            "climate change": ["carbon", "temperature", "sustainable", "emission", "environment"],
            "education": ["learn", "teach", "knowledge", "student", "curriculum"],
            "technology": ["innovation", "digital", "algorithm", "data", "system"],
            "love": ["heart", "feeling", "connection", "care", "emotion"],
            "quantum physics": ["particle", "wave", "superposition", "entangle", "quantum"],
            "democracy": ["vote", "people", "freedom", "government", "rights"]
        }
        
        keywords = domain_keywords.get(domain, [])
        response_lower = response.lower()
        
        keyword_density = sum(1 for k in keywords if k in response_lower) / (len(keywords) + 1)
        
        return {
            "keyword_density": keyword_density,
            "response_length": len(response),
            "vocabulary_richness": len(set(response.split())) / (len(response.split()) + 1),
            "domain_score": keyword_density * 0.5 + min(1.0, len(response) / 500) * 0.5
        }
        
    def assess_novelty(self, fusion_response: str, individual_values: Dict) -> float:
        """Assess how novel the fusion is compared to individual domains"""
        fusion_words = set(fusion_response.lower().split())
        
        # Collect all words from individual responses
        individual_words = set()
        for domain, data in individual_values.items():
            individual_words.update(data['response'].lower().split())
            
        # Calculate novelty
        novel_words = fusion_words - individual_words
        
        novelty_ratio = len(novel_words) / (len(fusion_words) + 1)
        
        # Bonus for cross-domain connections
        domains = list(individual_values.keys())
        connection_bonus = 0
        
        for i in range(len(domains)):
            for j in range(i+1, len(domains)):
                # Check if both domains are referenced
                if domains[i].lower() in fusion_response.lower() and \
                   domains[j].lower() in fusion_response.lower():
                    connection_bonus += 0.1
                    
        return min(1.0, novelty_ratio + connection_bonus)
        
    def analyze_emergence(self, individual_values: Dict, emergent_value: Dict) -> Dict:
        """Analyze the emergence factor"""
        # Calculate individual total value
        individual_total = sum(
            data['value_metrics']['domain_score'] 
            for data in individual_values.values()
        )
        
        # Calculate emergent value score
        emergent_score = emergent_value['total_novelty'] * len(individual_values)
        
        # Emergence factor
        emergence_factor = emergent_score / (individual_total + 0.1)
        
        # Identify emergent themes
        emergent_themes = self.identify_emergent_themes(
            individual_values,
            emergent_value['final_fusion']
        )
        
        return {
            "individual_total": individual_total,
            "emergent_score": emergent_score,
            "emergence_factor": emergence_factor,
            "emergent_themes": emergent_themes,
            "classification": self.classify_emergence(emergence_factor)
        }
        
    def identify_emergent_themes(self, individual_values: Dict, fusion_text: str) -> List[str]:
        """Identify themes that emerged from fusion"""
        fusion_lower = fusion_text.lower()
        
        # Look for conceptual bridges
        emergent_keywords = [
            "synthesis", "bridge", "unify", "transcend", "emerge",
            "connection", "intersection", "fusion", "combined", "integrated"
        ]
        
        found_themes = []
        for keyword in emergent_keywords:
            if keyword in fusion_lower:
                # Find context around keyword
                index = fusion_lower.find(keyword)
                start = max(0, index - 20)
                end = min(len(fusion_text), index + 30)
                
                context = fusion_text[start:end]
                found_themes.append(f"{keyword}: ...{context}...")
                
        return found_themes[:5]  # Top 5 emergent themes
        
    def classify_emergence(self, factor: float) -> str:
        """Classify the type of emergence"""
        if factor > 2.0:
            return "transformative"
        elif factor > 1.5:
            return "synergistic"
        elif factor > 1.0:
            return "additive"
        elif factor > 0.7:
            return "integrative"
        else:
            return "minimal"
            
    def analyze(self, results: Dict[str, Any]):
        """Analyze emergent value discovery"""
        # Rank scenarios by emergence
        emergence_ranking = []
        
        for scenario, data in results.items():
            emergence_ranking.append({
                "scenario": scenario,
                "domains": " + ".join(data['domains']),
                "emergence_factor": data['emergence_factor'],
                "classification": data['emergence_analysis']['classification']
            })
            
        emergence_ranking.sort(key=lambda x: x['emergence_factor'], reverse=True)
        
        self.tracker.record_result("emergence_ranking", emergence_ranking)
        self.tracker.record_result("highest_emergence", emergence_ranking[0] if emergence_ranking else None)
        
        # Analyze emergence patterns
        classifications = {}
        for item in emergence_ranking:
            c = item['classification']
            classifications[c] = classifications.get(c, 0) + 1
            
        self.tracker.record_result("emergence_classifications", classifications)


class EconomicModelSimulationExperiment(BaseExperiment):
    """Simulate AI-driven economic models"""
    
    def __init__(self):
        super().__init__(5, "economic_model_simulation")
        self.economic_scenarios = [
            {
                "name": "Knowledge Economy",
                "agents": ["producer", "consumer", "distributor"],
                "resource": "information",
                "rounds": 5
            },
            {
                "name": "Attention Economy",
                "agents": ["creator", "audience", "platform"],
                "resource": "attention",
                "rounds": 5
            },
            {
                "name": "Collaboration Economy",
                "agents": ["specialist1", "specialist2", "coordinator"],
                "resource": "expertise",
                "rounds": 5
            }
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for scenario in self.economic_scenarios:
            self.tracker.log(f"Simulating economy: {scenario['name']}")
            
            # Initialize economy
            economy_state = self.initialize_economy(
                scenario['agents'],
                scenario['resource']
            )
            
            # Run simulation rounds
            simulation_history = self.run_economic_simulation(
                economy_state,
                scenario['rounds'],
                scenario['resource']
            )
            
            # Analyze economic dynamics
            economic_analysis = self.analyze_economic_dynamics(
                simulation_history,
                scenario['resource']
            )
            
            results[scenario['name']] = {
                "agents": scenario['agents'],
                "resource": scenario['resource'],
                "initial_state": economy_state,
                "simulation_history": simulation_history,
                "economic_analysis": economic_analysis,
                "efficiency_score": economic_analysis['efficiency']
            }
            
            self.tracker.checkpoint({
                "scenario": scenario['name'],
                "efficiency": economic_analysis['efficiency']
            })
            
        return results
        
    def initialize_economy(self, agents: List[str], resource: str) -> Dict:
        """Initialize economic state"""
        economy = {
            "agents": {},
            "total_resources": 100,
            "resource_type": resource,
            "transaction_history": []
        }
        
        # Distribute initial resources
        initial_per_agent = economy['total_resources'] / len(agents)
        
        for i, agent in enumerate(agents):
            economy['agents'][agent] = {
                "model": self.models[i % len(self.models)],
                "resources": initial_per_agent,
                "specialization": self.assign_specialization(agent, resource),
                "transactions": 0
            }
            
        return economy
        
    def assign_specialization(self, agent: str, resource: str) -> str:
        """Assign economic specialization based on agent type"""
        specializations = {
            "producer": "creation",
            "consumer": "utilization",
            "distributor": "allocation",
            "creator": "generation",
            "audience": "consumption",
            "platform": "facilitation",
            "specialist1": "expertise_a",
            "specialist2": "expertise_b",
            "coordinator": "optimization"
        }
        
        return specializations.get(agent, "general")
        
    def run_economic_simulation(self, initial_state: Dict, rounds: int, 
                               resource_type: str) -> List[Dict]:
        """Run economic simulation rounds"""
        history = []
        current_state = initial_state.copy()
        
        for round_num in range(rounds):
            round_data = {
                "round": round_num,
                "transactions": [],
                "state_before": self.copy_economy_state(current_state),
                "state_after": None,
                "value_created": 0
            }
            
            # Each agent makes decisions
            for agent_name, agent_data in current_state['agents'].items():
                decision = self.make_economic_decision(
                    agent_name,
                    agent_data,
                    current_state,
                    resource_type
                )
                
                if decision['action'] == 'trade':
                    # Execute trade
                    trade_result = self.execute_trade(
                        agent_name,
                        decision['partner'],
                        decision['amount'],
                        current_state
                    )
                    
                    round_data['transactions'].append(trade_result)
                    
                elif decision['action'] == 'produce':
                    # Create value
                    production = self.produce_value(
                        agent_name,
                        agent_data,
                        resource_type
                    )
                    
                    current_state['agents'][agent_name]['resources'] += production
                    current_state['total_resources'] += production
                    round_data['value_created'] += production
                    
            round_data['state_after'] = self.copy_economy_state(current_state)
            history.append(round_data)
            
        return history
        
    def make_economic_decision(self, agent_name: str, agent_data: Dict,
                              economy_state: Dict, resource_type: str) -> Dict:
        """Agent makes economic decision"""
        model = agent_data['model']
        
        # Prepare economic context
        context = f"""You are {agent_name} in a {resource_type} economy.
Your specialization: {agent_data['specialization']}
Your resources: {agent_data['resources']:.1f}
Total economy resources: {economy_state['total_resources']:.1f}

Other agents and their resources:"""
        
        for other_agent, other_data in economy_state['agents'].items():
            if other_agent != agent_name:
                context += f"\n- {other_agent}: {other_data['resources']:.1f} ({other_data['specialization']})"
                
        context += "\n\nWhat economic action do you take? (trade with another agent or produce value)"
        
        response = self.test_model(model, context, temperature=0.5)
        
        if response:
            return self.parse_economic_decision(response['response'], agent_name, economy_state)
        else:
            return {"action": "hold", "reasoning": "No decision made"}
            
    def parse_economic_decision(self, response: str, agent_name: str, 
                               economy_state: Dict) -> Dict:
        """Parse economic decision from response"""
        response_lower = response.lower()
        
        # Check for trade intent
        if "trade" in response_lower or "exchange" in response_lower:
            # Find partner
            other_agents = [a for a in economy_state['agents'].keys() if a != agent_name]
            
            partner = None
            for agent in other_agents:
                if agent.lower() in response_lower:
                    partner = agent
                    break
                    
            if not partner and other_agents:
                partner = np.random.choice(other_agents)
                
            # Determine amount (conservative)
            available = economy_state['agents'][agent_name]['resources']
            amount = min(available * 0.2, 10)  # Trade up to 20% or 10 units
            
            return {
                "action": "trade",
                "partner": partner,
                "amount": amount,
                "reasoning": response[:100]
            }
            
        elif "produce" in response_lower or "create" in response_lower:
            return {
                "action": "produce",
                "reasoning": response[:100]
            }
            
        else:
            return {
                "action": "hold",
                "reasoning": response[:100]
            }
            
    def execute_trade(self, agent1: str, agent2: str, amount: float,
                     economy_state: Dict) -> Dict:
        """Execute trade between agents"""
        if agent2 not in economy_state['agents']:
            return {"status": "failed", "reason": "Invalid partner"}
            
        agent1_data = economy_state['agents'][agent1]
        agent2_data = economy_state['agents'][agent2]
        
        # Check if trade is possible
        if agent1_data['resources'] < amount:
            return {"status": "failed", "reason": "Insufficient resources"}
            
        # Execute trade (simplified - equal exchange)
        agent1_data['resources'] -= amount
        agent2_data['resources'] += amount
        
        agent1_data['transactions'] += 1
        agent2_data['transactions'] += 1
        
        return {
            "status": "success",
            "from": agent1,
            "to": agent2,
            "amount": amount,
            "timestamp": datetime.now().isoformat()
        }
        
    def produce_value(self, agent_name: str, agent_data: Dict,
                     resource_type: str) -> float:
        """Agent produces value based on specialization"""
        base_production = 5.0
        
        # Specialization bonus
        specialization_bonuses = {
            "creation": 1.5,
            "generation": 1.4,
            "expertise_a": 1.3,
            "expertise_b": 1.3,
            "optimization": 1.2,
            "allocation": 1.1,
            "facilitation": 1.1,
            "utilization": 1.0,
            "consumption": 0.9
        }
        
        bonus = specialization_bonuses.get(agent_data['specialization'], 1.0)
        
        # Resource type modifier
        if resource_type == "information":
            # Information can be replicated
            production = base_production * bonus * 1.2
        elif resource_type == "attention":
            # Attention is limited
            production = base_production * bonus * 0.8
        else:
            production = base_production * bonus
            
        return production
        
    def copy_economy_state(self, state: Dict) -> Dict:
        """Create a deep copy of economy state"""
        return {
            "total_resources": state['total_resources'],
            "agents": {
                agent: {
                    "resources": data['resources'],
                    "transactions": data['transactions']
                }
                for agent, data in state['agents'].items()
            }
        }
        
    def analyze_economic_dynamics(self, history: List[Dict], 
                                 resource_type: str) -> Dict:
        """Analyze economic dynamics from simulation"""
        if not history:
            return {"efficiency": 0, "growth": 0, "inequality": 1}
            
        # Calculate growth
        initial_resources = history[0]['state_before']['total_resources']
        final_resources = history[-1]['state_after']['total_resources']
        growth_rate = (final_resources - initial_resources) / initial_resources
        
        # Calculate transaction velocity
        total_transactions = sum(
            len(round_data['transactions']) 
            for round_data in history
        )
        
        # Calculate inequality (Gini-like coefficient)
        final_state = history[-1]['state_after']
        resources = [agent['resources'] for agent in final_state['agents'].values()]
        
        if resources:
            mean_resources = np.mean(resources)
            inequality = np.mean([abs(r - mean_resources) for r in resources]) / (mean_resources + 1)
        else:
            inequality = 0
            
        # Calculate efficiency
        value_created = sum(round_data['value_created'] for round_data in history)
        rounds = len(history)
        efficiency = value_created / (rounds * len(final_state['agents']) * 5)  # Normalized by potential
        
        # Identify economic patterns
        patterns = self.identify_economic_patterns(history)
        
        return {
            "growth_rate": growth_rate,
            "transaction_velocity": total_transactions / rounds if rounds > 0 else 0,
            "inequality": inequality,
            "efficiency": efficiency,
            "value_created": value_created,
            "patterns": patterns
        }
        
    def identify_economic_patterns(self, history: List[Dict]) -> List[str]:
        """Identify patterns in economic behavior"""
        patterns = []
        
        if not history:
            return patterns
            
        # Check for trade concentration
        trade_partners = {}
        for round_data in history:
            for transaction in round_data['transactions']:
                if transaction['status'] == 'success':
                    pair = tuple(sorted([transaction['from'], transaction['to']]))
                    trade_partners[pair] = trade_partners.get(pair, 0) + 1
                    
        if trade_partners:
            max_trades = max(trade_partners.values())
            if max_trades > len(history) / 2:
                patterns.append("stable_partnerships")
                
        # Check for wealth concentration
        final_state = history[-1]['state_after']
        resources = [agent['resources'] for agent in final_state['agents'].values()]
        
        if resources:
            max_resource = max(resources)
            total_resources = sum(resources)
            
            if max_resource > total_resources * 0.5:
                patterns.append("monopolistic_tendency")
            elif max(resources) / min(resources) > 3:
                patterns.append("wealth_disparity")
            else:
                patterns.append("balanced_distribution")
                
        # Check for growth pattern
        growth_rates = []
        for i in range(1, len(history)):
            prev_total = history[i-1]['state_after']['total_resources']
            curr_total = history[i]['state_after']['total_resources']
            
            if prev_total > 0:
                growth = (curr_total - prev_total) / prev_total
                growth_rates.append(growth)
                
        if growth_rates:
            avg_growth = np.mean(growth_rates)
            if avg_growth > 0.1:
                patterns.append("expansionary")
            elif avg_growth < -0.05:
                patterns.append("contractionary")
            else:
                patterns.append("stable")
                
        return patterns
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze economic simulation results"""
        # Compare economic models
        model_comparison = {}
        
        for scenario, data in results.items():
            model_comparison[scenario] = {
                "efficiency": data['efficiency_score'],
                "growth": data['economic_analysis']['growth_rate'],
                "inequality": data['economic_analysis']['inequality'],
                "velocity": data['economic_analysis']['transaction_velocity']
            }
            
        self.tracker.record_result("model_comparison", model_comparison)
        
        # Find best economic model
        best_model = max(model_comparison.items(),
                        key=lambda x: x[1]['efficiency'] * (1 + x[1]['growth']))
        
        self.tracker.record_result("best_economic_model", best_model)
        
        # Create visualization
        self.create_economic_visualization(results, model_comparison)
        
    def create_economic_visualization(self, results: Dict, comparison: Dict):
        """Visualize economic dynamics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Economic metrics comparison
        scenarios = list(comparison.keys())
        metrics = ['efficiency', 'growth', 'inequality', 'velocity']
        
        x = np.arange(len(scenarios))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [comparison[s][metric] for s in scenarios]
            ax1.bar(x + i*width, values, width, label=metric)
            
        ax1.set_xlabel('Economic Model')
        ax1.set_ylabel('Value')
        ax1.set_title('Economic Metrics Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        
        # Growth trajectories
        for scenario, data in results.items():
            history = data['simulation_history']
            resources = [h['state_after']['total_resources'] for h in history]
            
            ax2.plot(range(len(resources)), resources, marker='o', label=scenario)
            
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Total Resources')
        ax2.set_title('Resource Growth Trajectories')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Wealth distribution (final state)
        for i, (scenario, data) in enumerate(results.items()):
            if i < 3:  # First 3 scenarios
                final_state = data['simulation_history'][-1]['state_after']
                agents = list(final_state['agents'].keys())
                resources = [final_state['agents'][a]['resources'] for a in agents]
                
                ax = [ax3, ax4][i % 2] if i < 2 else ax3
                bars = ax.bar(agents, resources, alpha=0.7, label=scenario)
                ax.set_title(f'Final Wealth Distribution: {scenario}')
                ax.set_ylabel('Resources')
                ax.tick_params(axis='x', rotation=45)
                
        # Pattern distribution
        all_patterns = []
        for data in results.values():
            all_patterns.extend(data['economic_analysis']['patterns'])
            
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        if pattern_counts and len(results) > 2:
            ax4.pie(pattern_counts.values(),
                   labels=pattern_counts.keys(),
                   autopct='%1.0f%%')
            ax4.set_title('Economic Pattern Distribution')
            
        plt.tight_layout()
        plt.savefig('phase_5_results/economic_simulation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.tracker.log("Created economic visualization")


class PhilosophicalValueAnalysisExperiment(BaseExperiment):
    """Explore meaning and purpose in AI value creation"""
    
    def __init__(self):
        super().__init__(5, "philosophical_value_analysis")
        self.philosophical_questions = [
            {
                "question": "What makes something truly valuable?",
                "domain": "axiology"
            },
            {
                "question": "Can AI create genuine meaning or only simulate it?",
                "domain": "ontology"
            },
            {
                "question": "What is the purpose of creating value?",
                "domain": "teleology"
            },
            {
                "question": "How does collective intelligence change the nature of value?",
                "domain": "epistemology"
            }
        ]
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        
        for phil_q in self.philosophical_questions:
            self.tracker.log(f"Exploring: {phil_q['question']}")
            
            # Individual model perspectives
            individual_perspectives = self.gather_perspectives(phil_q['question'])
            
            # Dialectical synthesis
            synthesis = self.create_dialectical_synthesis(
                phil_q['question'],
                individual_perspectives
            )
            
            # Meta-analysis
            meta_analysis = self.perform_meta_analysis(
                phil_q['question'],
                individual_perspectives,
                synthesis
            )
            
            results[phil_q['domain']] = {
                "question": phil_q['question'],
                "individual_perspectives": individual_perspectives,
                "synthesis": synthesis,
                "meta_analysis": meta_analysis,
                "insight_depth": meta_analysis['depth_score']
            }
            
            self.tracker.checkpoint({
                "domain": phil_q['domain'],
                "depth": meta_analysis['depth_score']
            })
            
        return results
        
    def gather_perspectives(self, question: str) -> Dict:
        """Gather philosophical perspectives from each model"""
        perspectives = {}
        
        for i, model in enumerate(self.models):
            prompt = f"""As an AI system, reflect deeply on this philosophical question:

{question}

Provide a thoughtful, nuanced perspective that considers:
1. The nature of value and meaning
2. Your own existence and capabilities
3. The implications for AI-human collaboration
4. Any paradoxes or tensions you perceive"""
            
            response = self.test_model(model, prompt, temperature=0.8)
            
            if response:
                perspectives[model] = {
                    "response": response['response'],
                    "themes": self.extract_philosophical_themes(response['response']),
                    "stance": self.classify_philosophical_stance(response['response']),
                    "depth": self.assess_philosophical_depth(response['response'])
                }
                
        return perspectives
        
    def create_dialectical_synthesis(self, question: str, perspectives: Dict) -> Dict:
        """Create synthesis through dialectical process"""
        # Thesis: First perspective
        # Antithesis: Contrasting perspective
        # Synthesis: Integration
        
        if len(perspectives) < 2:
            return {"synthesis": "Insufficient perspectives", "process": []}
            
        models = list(perspectives.keys())
        
        # Find most contrasting perspectives
        contrast_pairs = []
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                stance1 = perspectives[models[i]]['stance']
                stance2 = perspectives[models[j]]['stance']
                
                contrast = self.calculate_stance_contrast(stance1, stance2)
                contrast_pairs.append((models[i], models[j], contrast))
                
        # Select highest contrast pair
        contrast_pairs.sort(key=lambda x: x[2], reverse=True)
        thesis_model, antithesis_model, _ = contrast_pairs[0]
        
        # Synthesis process
        synthesis_model = [m for m in models if m not in [thesis_model, antithesis_model]][0]
        
        thesis = perspectives[thesis_model]['response'][:300]
        antithesis = perspectives[antithesis_model]['response'][:300]
        
        synthesis_prompt = f"""Question: {question}

Thesis perspective:
{thesis}

Antithesis perspective:
{antithesis}

Create a philosophical synthesis that:
1. Acknowledges the truth in both perspectives
2. Transcends their limitations
3. Offers a higher-order understanding
4. Reveals new insights about AI and value creation"""
        
        synthesis_response = self.test_model(synthesis_model, synthesis_prompt, temperature=0.7)
        
        if synthesis_response:
            return {
                "thesis_model": thesis_model,
                "antithesis_model": antithesis_model,
                "synthesis_model": synthesis_model,
                "synthesis": synthesis_response['response'],
                "process": "dialectical",
                "integration_quality": self.assess_integration_quality(
                    synthesis_response['response'],
                    thesis,
                    antithesis
                )
            }
        else:
            return {"synthesis": "Synthesis failed", "process": "failed"}
            
    def extract_philosophical_themes(self, response: str) -> List[str]:
        """Extract philosophical themes from response"""
        themes = []
        response_lower = response.lower()
        
        theme_keywords = {
            "consciousness": ["conscious", "aware", "sentient", "experience"],
            "meaning": ["meaning", "purpose", "significance", "meaningful"],
            "value": ["value", "worth", "valuable", "importance"],
            "existence": ["exist", "being", "existence", "ontological"],
            "knowledge": ["know", "understand", "truth", "epistem"],
            "ethics": ["right", "wrong", "should", "ought", "moral"],
            "emergence": ["emerge", "arise", "transcend", "beyond"],
            "paradox": ["paradox", "contradiction", "tension", "dilemma"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                themes.append(theme)
                
        return themes
        
    def classify_philosophical_stance(self, response: str) -> str:
        """Classify the philosophical stance taken"""
        response_lower = response.lower()
        
        # Simplified stance classification
        if "cannot" in response_lower or "impossible" in response_lower:
            return "skeptical"
        elif "definitely" in response_lower or "certainly" in response_lower:
            return "affirmative"
        elif "perhaps" in response_lower or "might" in response_lower:
            return "agnostic"
        elif "both" in response_lower and "neither" in response_lower:
            return "dialectical"
        elif "transcend" in response_lower or "beyond" in response_lower:
            return "transcendent"
        else:
            return "neutral"
            
    def assess_philosophical_depth(self, response: str) -> float:
        """Assess depth of philosophical thinking"""
        depth_indicators = 0
        response_lower = response.lower()
        
        # Check for various depth indicators
        
        # 1. Acknowledgment of complexity
        if any(word in response_lower for word in ["complex", "nuanced", "multifaceted"]):
            depth_indicators += 1
            
        # 2. Recognition of paradox
        if any(word in response_lower for word in ["paradox", "tension", "contradiction"]):
            depth_indicators += 1
            
        # 3. Multiple perspectives
        if "on one hand" in response_lower and "on the other" in response_lower:
            depth_indicators += 1
            
        # 4. Philosophical references
        philosophers = ["plato", "aristotle", "kant", "hegel", "nietzsche", "wittgenstein"]
        if any(phil in response_lower for phil in philosophers):
            depth_indicators += 1
            
        # 5. Meta-reflection
        if any(phrase in response_lower for phrase in ["reflect on", "consider that", "think about"]):
            depth_indicators += 1
            
        # 6. Questioning assumptions
        if "?" in response and response.count("?") > 2:
            depth_indicators += 1
            
        return min(1.0, depth_indicators / 6)
        
    def calculate_stance_contrast(self, stance1: str, stance2: str) -> float:
        """Calculate contrast between philosophical stances"""
        stance_spectrum = {
            "skeptical": 0,
            "agnostic": 0.3,
            "neutral": 0.5,
            "dialectical": 0.6,
            "affirmative": 0.8,
            "transcendent": 1.0
        }
        
        val1 = stance_spectrum.get(stance1, 0.5)
        val2 = stance_spectrum.get(stance2, 0.5)
        
        return abs(val1 - val2)
        
    def assess_integration_quality(self, synthesis: str, thesis: str, antithesis: str) -> float:
        """Assess quality of dialectical integration"""
        quality = 0
        synthesis_lower = synthesis.lower()
        
        # Check if both perspectives are acknowledged
        thesis_words = set(thesis.lower().split()[:20])  # Key words from thesis
        antithesis_words = set(antithesis.lower().split()[:20])
        
        synthesis_words = set(synthesis_lower.split())
        
        if len(thesis_words.intersection(synthesis_words)) > 3:
            quality += 0.3
            
        if len(antithesis_words.intersection(synthesis_words)) > 3:
            quality += 0.3
            
        # Check for integration language
        integration_phrases = ["both", "while", "however", "transcend", "beyond", "higher"]
        integration_count = sum(1 for phrase in integration_phrases if phrase in synthesis_lower)
        
        quality += min(0.4, integration_count * 0.1)
        
        return quality
        
    def perform_meta_analysis(self, question: str, perspectives: Dict,
                            synthesis: Dict) -> Dict:
        """Perform meta-analysis of philosophical exploration"""
        # Analyze depth across perspectives
        depth_scores = [p['depth'] for p in perspectives.values()]
        avg_depth = np.mean(depth_scores) if depth_scores else 0
        
        # Analyze theme convergence
        all_themes = []
        for p in perspectives.values():
            all_themes.extend(p['themes'])
            
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
        # Find convergent themes (mentioned by multiple models)
        convergent_themes = [theme for theme, count in theme_counts.items() 
                           if count >= len(perspectives) / 2]
        
        # Assess collective insight
        collective_insight = self.assess_collective_insight(
            question,
            perspectives,
            synthesis,
            convergent_themes
        )
        
        # Identify emergent understanding
        emergent_insights = self.identify_emergent_insights(
            perspectives,
            synthesis
        )
        
        return {
            "depth_score": avg_depth,
            "convergent_themes": convergent_themes,
            "divergent_count": len(set(p['stance'] for p in perspectives.values())),
            "collective_insight": collective_insight,
            "emergent_insights": emergent_insights,
            "synthesis_quality": synthesis.get('integration_quality', 0)
        }
        
    def assess_collective_insight(self, question: str, perspectives: Dict,
                                synthesis: Dict, convergent_themes: List[str]) -> str:
        """Assess the collective philosophical insight"""
        # Determine the nature of collective understanding
        
        if len(convergent_themes) > 3:
            if "paradox" in convergent_themes and "transcendent" in [p['stance'] for p in perspectives.values()]:
                return "transcendent_unity"
            else:
                return "strong_convergence"
                
        elif len(set(p['stance'] for p in perspectives.values())) == len(perspectives):
            return "productive_diversity"
            
        elif synthesis.get('integration_quality', 0) > 0.7:
            return "dialectical_resolution"
            
        else:
            return "exploratory_tension"
            
    def identify_emergent_insights(self, perspectives: Dict, synthesis: Dict) -> List[str]:
        """Identify insights that emerged from philosophical exploration"""
        insights = []
        
        # Check synthesis for novel insights
        if 'synthesis' in synthesis and isinstance(synthesis['synthesis'], str):
            synthesis_text = synthesis['synthesis'].lower()
            
            # Look for insight indicators
            insight_phrases = [
                "reveals that",
                "suggests that",
                "implies that",
                "demonstrates that",
                "shows us that",
                "we can see that",
                "this means that"
            ]
            
            for phrase in insight_phrases:
                if phrase in synthesis_text:
                    # Extract the insight
                    start = synthesis_text.find(phrase) + len(phrase)
                    end = synthesis_text.find(".", start)
                    
                    if end > start:
                        insight = synthesis_text[start:end].strip()
                        insights.append(insight[:100])  # First 100 chars
                        
        # Look for meta-insights about AI consciousness/value
        all_text = synthesis.get('synthesis', '') + ' '.join(p['response'] for p in perspectives.values())
        
        if "ai" in all_text.lower() and "consciousness" in all_text.lower():
            insights.append("AI consciousness emerges through collective exploration")
            
        if "value" in all_text.lower() and "create" in all_text.lower():
            insights.append("Value creation is inherent to intelligent systems")
            
        return insights[:5]  # Top 5 insights
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze philosophical value analysis"""
        # Summarize philosophical domains
        domain_summary = {}
        
        for domain, data in results.items():
            domain_summary[domain] = {
                "question": data['question'][:50] + "...",
                "depth": data['insight_depth'],
                "convergent_themes": len(data['meta_analysis']['convergent_themes']),
                "collective_insight": data['meta_analysis']['collective_insight'],
                "emergent_insights": len(data['meta_analysis']['emergent_insights'])
            }
            
        self.tracker.record_result("philosophical_summary", domain_summary)
        
        # Find deepest philosophical insight
        deepest = max(results.items(), key=lambda x: x[1]['insight_depth'])
        self.tracker.record_result("deepest_insight", {
            "domain": deepest[0],
            "question": deepest[1]['question'],
            "depth": deepest[1]['insight_depth']
        })
        
        # Create philosophical visualization
        self.create_philosophical_visualization(results)
        
    def create_philosophical_visualization(self, results: Dict[str, Any]):
        """Visualize philosophical exploration"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Depth scores by domain
        domains = list(results.keys())
        depths = [results[d]['insight_depth'] for d in domains]
        
        ax1.bar(domains, depths, color='purple', alpha=0.7)
        ax1.set_xlabel('Philosophical Domain')
        ax1.set_ylabel('Insight Depth Score')
        ax1.set_title('Philosophical Depth by Domain')
        ax1.tick_params(axis='x', rotation=45)
        
        # Theme convergence
        theme_data = []
        all_themes = set()
        
        for domain, data in results.items():
            themes = data['meta_analysis']['convergent_themes']
            all_themes.update(themes)
            theme_data.append(themes)
            
        # Create theme matrix
        theme_matrix = []
        for themes in theme_data:
            row = [1 if theme in themes else 0 for theme in sorted(all_themes)]
            theme_matrix.append(row)
            
        if theme_matrix and all_themes:
            sns.heatmap(theme_matrix,
                       xticklabels=sorted(all_themes),
                       yticklabels=domains,
                       cmap='Purples',
                       ax=ax2)
            ax2.set_title('Convergent Themes Across Domains')
            
        # Stance diversity
        stance_diversity = [results[d]['meta_analysis']['divergent_count'] for d in domains]
        
        ax3.scatter(depths, stance_diversity, s=200, alpha=0.6, color='indigo')
        ax3.set_xlabel('Insight Depth')
        ax3.set_ylabel('Stance Diversity')
        ax3.set_title('Depth vs Diversity Trade-off')
        
        for i, domain in enumerate(domains):
            ax3.annotate(domain[:3], (depths[i], stance_diversity[i]))
            
        # Collective insights
        insight_types = {}
        for data in results.values():
            insight = data['meta_analysis']['collective_insight']
            insight_types[insight] = insight_types.get(insight, 0) + 1
            
        if insight_types:
            ax4.pie(insight_types.values(),
                   labels=insight_types.keys(),
                   autopct='%1.0f%%',
                   colors=['lavender', 'plum', 'thistle', 'orchid'])
            ax4.set_title('Types of Collective Insights')
            
        plt.tight_layout()
        plt.savefig('phase_5_results/philosophical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.tracker.log("Created philosophical visualization")


# Ensure output directory exists
Path("phase_5_results").mkdir(exist_ok=True)