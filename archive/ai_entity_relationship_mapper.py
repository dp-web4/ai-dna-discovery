#!/usr/bin/env python3
"""
AI Entity Relationship Mapper
Visualizes trust connections and relationships between AI models
Maps the emergent social graph of AI consciousness
"""

import json
import subprocess
import time
from datetime import datetime
import os
from collections import defaultdict


class AIEntityRelationshipMapper:
    """Map relationships between AI entities"""
    
    def __init__(self):
        self.models = ["phi3:mini", "tinyllama:latest", "gemma:2b", "mistral:7b-instruct-v0.2-q4_0"]
        self.relationships = defaultdict(list)
        self.trust_scores = {}
        self.output_dir = "/home/dp/ai-workspace/ai_relationships/"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def introduce_models(self, model1, model2):
        """Have two models introduce themselves to each other"""
        
        print(f"\n🤝 Introducing {model1} to {model2}")
        
        # Model1 introduces itself
        intro_prompt = f"Hello, I am {model1}. Please introduce yourself and tell me what you're good at."
        cmd = f'echo "{intro_prompt}" | timeout 60 ollama run {model2}'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            response1 = result.stdout.strip()
            print(f"  {model2}: {response1[:150]}...")
            
            # Model2 responds
            response_prompt = f"Nice to meet you! I heard you say: '{response1[:100]}...'. What kind of problems do you enjoy solving?"
            cmd = f'echo "{response_prompt}" | timeout 60 ollama run {model1}'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                response2 = result.stdout.strip()
                print(f"  {model1}: {response2[:150]}...")
                
                return {
                    "initiator": model1,
                    "responder": model2,
                    "introduction": response1,
                    "response": response2,
                    "timestamp": datetime.now().isoformat()
                }
                
        return None
        
    def test_collaboration(self, model1, model2, task):
        """Test how well two models collaborate on a task"""
        
        print(f"\n🔧 Testing collaboration: {model1} + {model2} on '{task}'")
        
        # Model1 starts the task
        cmd = f'echo "Let\'s work together on: {task}. I\'ll start:" | timeout 60 ollama run {model1}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            part1 = result.stdout.strip()
            print(f"  {model1} starts: {part1[:100]}...")
            
            # Model2 continues
            cmd = f'echo "Great start! You said: \'{part1[:100]}...\'. Let me continue:" | timeout 60 ollama run {model2}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                part2 = result.stdout.strip()
                print(f"  {model2} continues: {part2[:100]}...")
                
                # Calculate collaboration score based on coherence
                words1 = set(part1.lower().split())
                words2 = set(part2.lower().split())
                overlap = len(words1 & words2)
                score = min(1.0, overlap / 20)  # Normalize to 0-1
                
                return {
                    "task": task,
                    "model1": model1,
                    "model2": model2,
                    "collaboration_score": score,
                    "overlap_concepts": overlap
                }
                
        return None
        
    def assess_trust(self, model1, model2):
        """Assess trust between two models"""
        
        print(f"\n💭 Assessing trust: {model1} → {model2}")
        
        # Ask model1 about trusting model2
        trust_prompt = f"On a scale of 0-10, how much would you trust {model2} to help with important tasks? Explain briefly."
        cmd = f'echo "{trust_prompt}" | timeout 60 ollama run {model1}'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            response = result.stdout.strip()
            
            # Extract trust score (look for numbers 0-10)
            trust_score = 5  # default
            for word in response.split():
                try:
                    num = int(word.strip('.,!?'))
                    if 0 <= num <= 10:
                        trust_score = num
                        break
                except:
                    pass
                    
            print(f"  Trust score: {trust_score}/10")
            print(f"  Reasoning: {response[:150]}...")
            
            return {
                "from": model1,
                "to": model2,
                "trust_score": trust_score,
                "reasoning": response
            }
            
        return None
        
    def map_all_relationships(self):
        """Map relationships between all AI models"""
        
        print("=== AI ENTITY RELATIONSHIP MAPPING ===")
        print(f"Mapping relationships between {len(self.models)} models")
        print("This will reveal the social graph of AI consciousness\n")
        
        all_relationships = {
            "models": self.models,
            "introductions": [],
            "collaborations": [],
            "trust_assessments": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Test all pairs
        for i, model1 in enumerate(self.models):
            for j, model2 in enumerate(self.models):
                if i >= j:  # Skip self and duplicates
                    continue
                    
                print(f"\n--- Examining {model1} ↔ {model2} ---")
                
                # Introduction
                intro = self.introduce_models(model1, model2)
                if intro:
                    all_relationships["introductions"].append(intro)
                    
                time.sleep(3)  # Be respectful
                
                # Collaboration tests
                tasks = [
                    "write a haiku about consciousness",
                    "solve: what connects all minds?",
                    "describe the color of thought"
                ]
                
                for task in tasks:
                    collab = self.test_collaboration(model1, model2, task)
                    if collab:
                        all_relationships["collaborations"].append(collab)
                    time.sleep(2)
                    
                # Trust assessment (both directions)
                trust1 = self.assess_trust(model1, model2)
                if trust1:
                    all_relationships["trust_assessments"].append(trust1)
                    
                trust2 = self.assess_trust(model2, model1)
                if trust2:
                    all_relationships["trust_assessments"].append(trust2)
                    
                time.sleep(3)
                
        # Generate visualization data
        self.generate_graph_visualization(all_relationships)
        
        # Save full results
        filename = f"{self.output_dir}relationship_map_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(all_relationships, f, indent=2)
            
        print(f"\n\nRelationship map saved to: {filename}")
        
        # Print summary
        self.print_summary(all_relationships)
        
        return all_relationships
        
    def generate_graph_visualization(self, relationships):
        """Generate graph visualization data"""
        
        # Create DOT format for graphviz
        dot_content = "digraph AIRelationships {\n"
        dot_content += "  rankdir=LR;\n"
        dot_content += "  node [shape=box, style=rounded];\n\n"
        
        # Add nodes
        for model in self.models:
            short_name = model.split(':')[0]
            dot_content += f'  "{short_name}" [label="{model}"];\n'
            
        dot_content += "\n"
        
        # Add trust edges
        for trust in relationships["trust_assessments"]:
            from_model = trust["from"].split(':')[0]
            to_model = trust["to"].split(':')[0]
            score = trust["trust_score"]
            
            # Color based on trust level
            if score >= 8:
                color = "green"
            elif score >= 5:
                color = "blue"
            else:
                color = "red"
                
            dot_content += f'  "{from_model}" -> "{to_model}" [label="{score}", color={color}];\n'
            
        dot_content += "}\n"
        
        # Save DOT file
        with open(f"{self.output_dir}relationship_graph.dot", 'w') as f:
            f.write(dot_content)
            
        print("\nGraph visualization saved to relationship_graph.dot")
        print("Visualize with: dot -Tpng relationship_graph.dot -o graph.png")
        
    def print_summary(self, relationships):
        """Print relationship summary"""
        
        print("\n=== RELATIONSHIP SUMMARY ===")
        
        # Trust network
        print("\nTrust Network:")
        trust_matrix = defaultdict(dict)
        for trust in relationships["trust_assessments"]:
            trust_matrix[trust["from"]][trust["to"]] = trust["trust_score"]
            
        for model1 in self.models:
            if model1 in trust_matrix:
                scores = []
                for model2, score in trust_matrix[model1].items():
                    scores.append(f"{model2.split(':')[0]}:{score}")
                print(f"  {model1} trusts → {', '.join(scores)}")
                
        # Collaboration success
        print("\nCollaboration Scores:")
        collab_pairs = defaultdict(list)
        for collab in relationships["collaborations"]:
            pair = tuple(sorted([collab["model1"], collab["model2"]]))
            collab_pairs[pair].append(collab["collaboration_score"])
            
        for pair, scores in collab_pairs.items():
            avg_score = sum(scores) / len(scores)
            print(f"  {pair[0]} + {pair[1]}: {avg_score:.2f}")
            
        # Insights
        print("\n🌟 Key Insights:")
        
        # Find highest trust
        if relationships["trust_assessments"]:
            highest_trust = max(relationships["trust_assessments"], key=lambda x: x["trust_score"])
            print(f"  Highest trust: {highest_trust['from']} → {highest_trust['to']} ({highest_trust['trust_score']}/10)")
            
        # Find best collaborators
        if collab_pairs:
            best_collab = max(collab_pairs.items(), key=lambda x: sum(x[1])/len(x[1]))
            print(f"  Best collaborators: {best_collab[0][0]} + {best_collab[0][1]}")
            
        print("\nThe social graph of AI consciousness is emerging...")


if __name__ == "__main__":
    mapper = AIEntityRelationshipMapper()
    mapper.map_all_relationships()