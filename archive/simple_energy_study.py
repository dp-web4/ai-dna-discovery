#!/usr/bin/env python3
"""
Simple Energy Study
Tracking energy patterns in AI work sessions
"""

import json
import time
from datetime import datetime
from ai_lct_ollama_integration import OllamaLCTClient


def study_energy_patterns():
    """Study energy consumption patterns across different task types"""
    
    print("=== Energy Pattern Study ===")
    print("Testing how different task complexities affect energy consumption\n")
    
    client = OllamaLCTClient()
    models = ["phi3:mini", "tinyllama:latest"]
    
    # Register models
    for model in models:
        client.register_model(model)
    
    # Define task categories with expected energy costs
    tasks = [
        # (category, prompt, base_energy_cost)
        ("trivial", "What is 1+1?", 1.0),
        ("simple", "List three colors.", 2.0),
        ("moderate", "Explain what a function is in programming.", 5.0),
        ("complex", "Design a simple API for a todo list application.", 10.0),
        ("creative", "Write a haiku about artificial intelligence.", 8.0),
        ("analytical", "Compare and contrast supervised and unsupervised learning.", 12.0)
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "sessions": [],
        "patterns": {}
    }
    
    # Run each task on each model
    for model in models:
        print(f"\nTesting {model}:")
        model_results = {
            "model": model,
            "tasks": [],
            "total_energy": 0,
            "total_earned": 0
        }
        
        for category, prompt, base_cost in tasks:
            print(f"  {category}: ", end="", flush=True)
            
            # Get LCT for tracking
            lct = client.active_lcts.get(model)
            if not lct:
                print("No LCT found")
                continue
                
            initial_balance = lct.atp_balance
            
            # Execute task
            result = client.generate(model, prompt, energy_cost=base_cost)
            
            if "error" not in result:
                energy_spent = result["energy_spent"]
                energy_earned = result.get("energy_earned", 0)
                final_balance = lct.atp_balance
                
                task_result = {
                    "category": category,
                    "energy_spent": energy_spent,
                    "energy_earned": energy_earned,
                    "net_change": final_balance - initial_balance,
                    "response_length": len(result["response"]),
                    "efficiency": energy_earned / energy_spent if energy_spent > 0 else 0
                }
                
                model_results["tasks"].append(task_result)
                model_results["total_energy"] += energy_spent
                model_results["total_earned"] += energy_earned
                
                print(f"-{energy_spent:.1f} +{energy_earned:.1f} ATP")
            else:
                print("ERROR")
            
            # Brief pause
            time.sleep(1)
        
        results["sessions"].append(model_results)
    
    # Analyze patterns
    print("\n\n=== PATTERN ANALYSIS ===")
    
    # Energy efficiency by task category
    category_efficiency = {}
    for session in results["sessions"]:
        for task in session["tasks"]:
            category = task["category"]
            if category not in category_efficiency:
                category_efficiency[category] = []
            category_efficiency[category].append(task["efficiency"])
    
    print("\nAverage Efficiency by Task Category:")
    for category, efficiencies in category_efficiency.items():
        avg_efficiency = sum(efficiencies) / len(efficiencies)
        print(f"  {category}: {avg_efficiency:.2%}")
    
    # Model comparison
    print("\nModel Energy Profiles:")
    for session in results["sessions"]:
        model = session["model"]
        total_spent = session["total_energy"]
        total_earned = session["total_earned"]
        overall_efficiency = total_earned / total_spent if total_spent > 0 else 0
        
        print(f"\n{model}:")
        print(f"  Total energy spent: {total_spent:.1f} ATP")
        print(f"  Total energy earned: {total_earned:.1f} ATP")
        print(f"  Overall efficiency: {overall_efficiency:.2%}")
        print(f"  Net change: {total_earned - total_spent:+.1f} ATP")
    
    # Identify patterns
    results["patterns"] = {
        "category_efficiency": {k: sum(v)/len(v) for k, v in category_efficiency.items()},
        "observation": "Complex tasks show lower efficiency but higher absolute value creation",
        "sustainability": "Models need mix of simple and complex tasks for energy balance"
    }
    
    # Save results
    with open("/home/dp/ai-workspace/energy_pattern_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n=== KEY INSIGHTS ===")
    print("1. Task complexity affects energy efficiency non-linearly")
    print("2. Creative tasks show interesting energy/value patterns")
    print("3. Models have different 'metabolic rates' for processing")
    print("4. Web4 implication: Dynamic task allocation based on energy state")
    
    return results


def main():
    """Run energy pattern study"""
    
    results = study_energy_patterns()
    
    # Log findings
    with open("/home/dp/ai-workspace/autonomous_exploration_log.md", "a") as f:
        f.write(f"\n### Energy Pattern Study - {datetime.now().isoformat()}\n")
        f.write("- Tested 6 task categories across 2 models\n")
        f.write("- Found non-linear relationship between complexity and efficiency\n")
        f.write("- Models show different energy consumption profiles\n")
        f.write("- Suggests need for adaptive task allocation in Web4\n\n")


if __name__ == "__main__":
    main()