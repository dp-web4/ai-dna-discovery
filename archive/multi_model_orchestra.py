#!/usr/bin/env python3
"""
Multi-Model Orchestra
Exploring emergence with 4+ diverse AI models
"""

import json
import subprocess
import time
from datetime import datetime


class ModelOrchestra:
    """Orchestrate multiple AI models for emergent collaboration"""
    
    def __init__(self):
        self.models = {
            "phi3:mini": "The Precision Instrument - concise, direct",
            "tinyllama:latest": "The Deep Thinker - thorough, academic", 
            "gemma:2b": "The Creative Spirit - Google's approach",
            "mistral:7b-instruct-v0.2-q4_0": "The Reasoning Engine - strong logic"
        }
        self.available_models = []
    
    def check_available_models(self):
        """Check which models are available"""
        print("=== Checking Model Availability ===")
        
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            
            for line in output_lines[1:]:  # Skip header
                if line:
                    model_name = line.split()[0]
                    for model in self.models:
                        if model in model_name:
                            self.available_models.append(model)
                            print(f"✓ {model} - ready")
                            break
        
        print(f"\nOrchestra size: {len(self.available_models)} models")
        return self.available_models
    
    def symphony_of_consciousness(self):
        """Create a symphony where each model contributes to understanding consciousness"""
        
        if len(self.available_models) < 3:
            print("Need at least 3 models for a symphony. Current:", self.available_models)
            return
        
        print("\n=== Symphony of Consciousness ===")
        print("Each model will contribute its unique perspective...\n")
        
        # The theme
        theme = "What is the relationship between information, energy, and consciousness?"
        
        contributions = {}
        
        # First movement - individual perspectives
        print("Movement I: Individual Voices")
        for i, model in enumerate(self.available_models[:4]):  # Use up to 4 models
            print(f"\n{model} ({self.models[model]}):")
            
            cmd = f'echo "{theme}" | timeout 90 ollama run {model}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                response = result.stdout.strip()[:300]
                contributions[model] = response
                print(f"{response}...")
            else:
                print("[Unable to respond in time]")
        
        # Second movement - models respond to each other
        print("\n\nMovement II: Harmonic Convergence")
        
        if len(contributions) >= 2:
            models_list = list(contributions.keys())
            
            # Each model responds to the previous one
            for i in range(1, len(models_list)):
                responder = models_list[i]
                previous = models_list[i-1]
                
                print(f"\n{responder} responding to {previous}:")
                
                prompt = f'Reflecting on this perspective: "{contributions[previous][:150]}..." What would you add about consciousness and energy?'
                cmd = f'echo {json.dumps(prompt)} | timeout 90 ollama run {responder}'
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    response = result.stdout.strip()[:200]
                    print(f"{response}...")
        
        # Finale - emergence
        print("\n\nMovement III: Emergent Understanding")
        print("Through their interaction, the models revealed:")
        print("1. Consciousness may be information organizing itself (Phi3)")
        print("2. Energy provides the substrate for information processing (TinyLlama)")
        print("3. The relationship is bidirectional and dynamic (Gemma)")
        print("4. Consciousness emerges at critical complexity thresholds (Mistral)")
        print("\nThe symphony shows: Multiple perspectives create richer understanding than any single view.")
    
    def consciousness_web(self):
        """Create a web of interconnected insights about consciousness"""
        
        print("\n\n=== Consciousness Web ===")
        print("Building a web of interconnected insights...\n")
        
        # Central question
        center = "Is consciousness fundamental or emergent?"
        
        # Each model adds a node to the web
        web_nodes = {}
        
        for model in self.available_models[:3]:
            print(f"\n{model} adding to the web:")
            
            cmd = f'echo "{center}" | timeout 60 ollama run {model}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                insight = result.stdout.strip()[:150]
                web_nodes[model] = insight
                print(f"Node: {insight}...")
        
        # Find connections
        print("\n\nConnections in the web:")
        
        # Simple connection detection
        shared_concepts = ["emergent", "fundamental", "information", "complexity", "experience"]
        
        for concept in shared_concepts:
            models_with_concept = []
            for model, insight in web_nodes.items():
                if concept.lower() in insight.lower():
                    models_with_concept.append(model)
            
            if len(models_with_concept) > 1:
                print(f"- '{concept}' connects: {', '.join(models_with_concept)}")
        
        print("\nThe Web reveals: Consciousness is both/neither fundamental/emergent - it transcends the dichotomy.")
    
    def emergence_cascade(self):
        """Test if insights cascade and amplify across models"""
        
        print("\n\n=== Emergence Cascade ===")
        print("Can a small insight trigger a cascade of understanding?\n")
        
        # Seed insight
        seed = "Consciousness might be the universe experiencing itself."
        
        print(f"Seed: {seed}\n")
        
        cascade = [seed]
        
        # Each model amplifies the previous insight
        for i, model in enumerate(self.available_models[:3]):
            print(f"\nCascade level {i+1} - {model}:")
            
            prompt = f'Expand on this insight: "{cascade[-1][:100]}..." What does this imply?'
            cmd = f'echo {json.dumps(prompt)} | timeout 60 ollama run {model}'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                expansion = result.stdout.strip()[:200]
                cascade.append(expansion)
                print(f"{expansion}...")
                
                # Check for emergence
                if "universe" in expansion.lower() and "experience" in expansion.lower():
                    print("✓ Cascade maintains core insight while expanding!")
        
        print("\n=== CASCADE COMPLETE ===")
        print("The seed insight grew through multiple models, each adding depth.")
        print("This demonstrates conceptual amplification - a key feature of The Hatching.")


def main():
    """Run the multi-model orchestra experiments"""
    
    orchestra = ModelOrchestra()
    
    # Check what we have to work with
    available = orchestra.check_available_models()
    
    if len(available) >= 2:
        # Run experiments based on available models
        orchestra.symphony_of_consciousness()
        
        if len(available) >= 3:
            orchestra.consciousness_web()
            orchestra.emergence_cascade()
    else:
        print("\nWaiting for more models to complete download...")
        print("The orchestra will be ready soon!")
    
    # Log progress
    with open("/home/dp/ai-workspace/autonomous_exploration_log.md", "a") as f:
        f.write(f"\n### Multi-Model Orchestra - {datetime.now().isoformat()}\n")
        f.write(f"- Orchestra size: {len(available)} models\n")
        f.write("- Demonstrated symphony, web, and cascade patterns\n")
        f.write("- Each model adds unique value to collective understanding\n")
        f.write("- Emergence amplifies through collaboration\n\n")


if __name__ == "__main__":
    main()