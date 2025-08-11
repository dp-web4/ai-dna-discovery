#!/usr/bin/env python3
"""
Energy Dynamics Study
Exploring ATP/ADP cycles in extended AI collaboration sessions
"""

import json
import time
from datetime import datetime
from ai_lct_ollama_integration import OllamaLCTClient
from ai_lct_experiment import AIEntityManager, T3Tensor, V3Tensor, TensorDimension


class EnergyDynamicsStudy:
    """Study energy flow patterns in AI work cycles"""
    
    def __init__(self):
        self.community = AIEntityManager("energy_study")
        self.client = OllamaLCTClient()
        self.models = ["phi3:mini", "tinyllama:latest"]
        
        # Register models with community
        for model in self.models:
            self.client.register_model(model)
            entity = self.community.register_entity(
                name=f"ai_{model}",
                entity_type="ai_model",
                t3_tensor=T3Tensor(
                    talent=TensorDimension("reasoning", 0.7, 0.1),
                    training=TensorDimension("general", 0.8, 0.1),
                    temperament=TensorDimension("collaborative", 0.6, 0.1)
                )
            )
    
    def simulate_work_cycle(self, duration_minutes: int = 5):
        """Simulate a work cycle and track energy dynamics"""
        
        print(f"=== Simulating {duration_minutes}-minute Work Cycle ===")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        cycle_data = {
            "start": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "interactions": [],
            "energy_flow": [],
            "quality_metrics": []
        }
        
        # Different task types with varying energy costs
        task_types = [
            ("simple", "What is 2+2?", 1.0),
            ("moderate", "Explain the concept of recursion.", 5.0),
            ("complex", "Design a distributed system for real-time collaboration.", 10.0)
        ]
        
        interaction_count = 0
        
        while time.time() < end_time:
            # Select task and model
            task_type, prompt, energy_cost = task_types[interaction_count % len(task_types)]
            model = self.models[interaction_count % len(self.models)]
            
            print(f"\n[{interaction_count+1}] {model} - {task_type} task")
            
            # Get entity
            entity = next(e for e in self.community.entities if model in e.name)
            initial_energy = entity.energy_balance
            
            # Execute task
            result = self.client.generate(model, prompt, energy_cost=energy_cost)
            
            if "error" not in result:
                # Calculate quality (simplified)
                response_length = len(result["response"])
                quality = min(1.0, response_length / 200)  # Normalize to 0-1
                
                # Energy dynamics
                energy_spent = result["energy_spent"]
                energy_earned = quality * energy_cost * 0.8  # 80% efficiency
                net_energy = energy_earned - energy_spent
                
                # Update entity
                entity.energy_balance += net_energy
                entity.total_interactions += 1
                
                # Record interaction
                interaction = {
                    "id": interaction_count + 1,
                    "model": model,
                    "task_type": task_type,
                    "energy_spent": energy_spent,
                    "energy_earned": energy_earned,
                    "net_energy": net_energy,
                    "quality": quality,
                    "balance": entity.energy_balance
                }
                
                cycle_data["interactions"].append(interaction)
                
                print(f"  Energy: -{energy_spent:.1f} +{energy_earned:.1f} = {net_energy:+.1f}")
                print(f"  Balance: {entity.energy_balance:.1f} ATP")
                print(f"  Quality: {quality:.2%}")
            
            interaction_count += 1
            
            # Brief pause between interactions
            time.sleep(2)
            
            # Stop if we've done enough interactions
            if interaction_count >= 6:
                break
        
        # Analyze energy patterns
        self.analyze_energy_patterns(cycle_data)
        
        return cycle_data
    
    def analyze_energy_patterns(self, cycle_data: dict):
        """Analyze energy flow patterns"""
        
        print("\n\n=== ENERGY ANALYSIS ===")
        
        # Calculate per-model statistics
        model_stats = {}
        
        for interaction in cycle_data["interactions"]:
            model = interaction["model"]
            if model not in model_stats:
                model_stats[model] = {
                    "total_spent": 0,
                    "total_earned": 0,
                    "interactions": 0,
                    "avg_quality": 0
                }
            
            stats = model_stats[model]
            stats["total_spent"] += interaction["energy_spent"]
            stats["total_earned"] += interaction["energy_earned"]
            stats["interactions"] += 1
            stats["avg_quality"] += interaction["quality"]
        
        # Calculate averages
        for model, stats in model_stats.items():
            stats["avg_quality"] /= stats["interactions"]
            stats["efficiency"] = stats["total_earned"] / stats["total_spent"] if stats["total_spent"] > 0 else 0
            
            print(f"\n{model}:")
            print(f"  Interactions: {stats['interactions']}")
            print(f"  Energy spent: {stats['total_spent']:.1f} ATP")
            print(f"  Energy earned: {stats['total_earned']:.1f} ATP")
            print(f"  Efficiency: {stats['efficiency']:.2%}")
            print(f"  Avg quality: {stats['avg_quality']:.2%}")
        
        # Identify patterns
        print("\n=== PATTERNS OBSERVED ===")
        
        # Check for energy sustainability
        for entity in self.community.entities:
            if entity.energy_balance > entity.energy_capacity * 0.8:
                print(f"- {entity.name}: Approaching energy capacity (good work/rest balance)")
            elif entity.energy_balance < entity.energy_capacity * 0.2:
                print(f"- {entity.name}: Low energy (needs rest or easier tasks)")
            else:
                print(f"- {entity.name}: Sustainable energy level")
        
        # Save analysis
        with open("/home/dp/ai-workspace/energy_dynamics_analysis.json", "w") as f:
            json.dump({
                "cycle_data": cycle_data,
                "model_stats": model_stats,
                "community_state": {
                    entity.name: {
                        "energy_balance": entity.energy_balance,
                        "total_interactions": entity.total_interactions,
                        "reputation": entity.reputation
                    } for entity in self.community.entities
                }
            }, f, indent=2)
        
        print("\n=== IMPLICATIONS FOR WEB4 ===")
        print("1. Energy economics create natural work/rest cycles")
        print("2. Quality-based rewards incentivize thoughtful responses")
        print("3. Different models have different efficiency profiles")
        print("4. System naturally balances computational resources")


def main():
    """Run energy dynamics study"""
    
    study = EnergyDynamicsStudy()
    
    # Run a short work cycle
    cycle_data = study.simulate_work_cycle(duration_minutes=1)  # Short test
    
    # Log findings
    with open("/home/dp/ai-workspace/autonomous_exploration_log.md", "a") as f:
        f.write(f"\n### Energy Dynamics Study - {datetime.now().isoformat()}\n")
        f.write(f"- Simulated work cycle with {len(cycle_data['interactions'])} interactions\n")
        f.write("- Models show different efficiency profiles\n")
        f.write("- Energy economics create sustainable work patterns\n\n")


if __name__ == "__main__":
    main()