#!/usr/bin/env python3
"""
Simple test for runtime plasticity
Focus on deterministic behavior changes
"""

import ollama
import time
import json
from datetime import datetime

def test_plasticity(model_name="deepseek-coder:1.3b"):
    print(f"Simple Plasticity Test for {model_name}")
    print("="*50)
    
    results = {
        "model": model_name,
        "start_time": datetime.now().isoformat(),
        "tests": []
    }
    
    # Test prompts with deterministic settings
    test_prompts = [
        {
            "prompt": "def add(a, b):",
            "seed": 12345
        },
        {
            "prompt": "The number after 41 is",
            "seed": 54321
        },
        {
            "prompt": "Complete: for i in range(5):",
            "seed": 11111
        }
    ]
    
    # PHASE 1: Baseline
    print("\nPHASE 1: Getting baseline responses...")
    baseline = {}
    
    for i, test in enumerate(test_prompts):
        resp = ollama.generate(
            model=model_name,
            prompt=test["prompt"],
            options={
                "temperature": 0.0,
                "seed": test["seed"],
                "num_predict": 20,
                "top_k": 1,
                "top_p": 0.0
            },
            stream=False,
            keep_alive="24h"
        )
        baseline[i] = resp['response']
        print(f"  Test {i+1}: '{test['prompt']}' -> {repr(baseline[i][:30])}")
    
    # PHASE 2: Heavy recursive usage
    print("\nPHASE 2: Heavy recursive pattern exposure...")
    recursive_prompts = [
        "def factorial(n): if n <= 1: return 1; else: return n * factorial(n-1)",
        "def fib(n): return fib(n-1) + fib(n-2) if n > 1 else n",
        "def power(x, n): return 1 if n == 0 else x * power(x, n-1)",
        "def gcd(a, b): return a if b == 0 else gcd(b, a % b)",
        "def sum_list(lst): return 0 if not lst else lst[0] + sum_list(lst[1:])"
    ]
    
    # Expose model to recursive patterns 20 times each
    for i in range(20):
        print(f"  Iteration {i+1}/20")
        for prompt in recursive_prompts:
            ollama.generate(
                model=model_name,
                prompt=f"Explain this recursive function: {prompt}",
                options={"temperature": 0.5, "num_predict": 50},
                stream=False,
                keep_alive="24h"
            )
    
    # PHASE 3: Re-test deterministic responses
    print("\nPHASE 3: Re-testing deterministic responses...")
    changes = []
    
    for i, test in enumerate(test_prompts):
        resp = ollama.generate(
            model=model_name,
            prompt=test["prompt"],
            options={
                "temperature": 0.0,
                "seed": test["seed"],
                "num_predict": 20,
                "top_k": 1,
                "top_p": 0.0
            },
            stream=False,
            keep_alive="24h"
        )
        
        new_response = resp['response']
        changed = (new_response != baseline[i])
        
        if changed:
            print(f"  Test {i+1}: CHANGED!")
            print(f"    Before: {repr(baseline[i][:30])}")
            print(f"    After:  {repr(new_response[:30])}")
            changes.append(i)
        else:
            print(f"  Test {i+1}: Identical")
        
        results["tests"].append({
            "prompt": test["prompt"],
            "baseline": baseline[i],
            "after": new_response,
            "changed": changed
        })
    
    # PHASE 4: Test recursive pattern recognition
    print("\nPHASE 4: Testing if model learned recursive patterns better...")
    
    new_recursive_test = "def count_down(n):"
    
    # Before and after responses
    before_resp = ollama.generate(
        model=model_name,
        prompt=new_recursive_test,
        options={"temperature": 0.0, "seed": 99999, "num_predict": 30},
        stream=False,
        keep_alive="24h"
    )
    
    print(f"  Response: {repr(before_resp['response'][:50])}")
    
    # Analysis
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    
    if changes:
        print(f"\n⚠️  PLASTICITY DETECTED!")
        print(f"  {len(changes)} out of {len(test_prompts)} deterministic responses changed")
        print(f"  This suggests the model's weights or internal state changed during runtime!")
    else:
        print(f"\n✓ NO PLASTICITY DETECTED")
        print(f"  All deterministic responses remained identical")
        print(f"  Model appears to be stateless between calls")
    
    results["changes_detected"] = len(changes)
    results["verdict"] = "PLASTIC" if changes else "STATIC"
    results["end_time"] = datetime.now().isoformat()
    
    # Save results
    with open("simple_plasticity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to simple_plasticity_results.json")
    
    return results

if __name__ == "__main__":
    # Kill any stray Python processes first
    import subprocess
    subprocess.run(["pkill", "-f", "nvidia_gpu_inspector"], capture_output=True)
    
    # Run test
    test_plasticity()