#!/usr/bin/env python3
"""
Quick version of pattern interpretation test
"""

import subprocess
import json
import time
import hashlib
import os
from datetime import datetime

class QuickPatternTest:
    def __init__(self):
        self.model_name = "phi3:mini"
        self.results_dir = "/home/dp/ai-workspace/ai-agents/phi3_pattern_quick"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def get_pattern_excerpt(self):
        """Get a meaningful excerpt from patterns.txt"""
        with open("/home/dp/ai-workspace/patterns.txt", 'r') as f:
            lines = f.readlines()
        
        # Get the elephant parable section (lines 1-45)
        excerpt = ''.join(lines[:45])
        return excerpt
    
    def query_model(self, prompt, temp=0.7, seed=None):
        """Query Phi3"""
        options = {"temperature": temp}
        if seed: options["seed"] = seed
        
        try:
            response = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True, text=True, timeout=30
            )
            return response.stdout.strip()
        except:
            return ""
    
    def run_test(self):
        """Run the pattern interpretation test"""
        print("PHI3 PATTERN INTERPRETATION - QUICK TEST")
        print("=" * 50)
        
        # Get excerpt
        excerpt = self.get_pattern_excerpt()
        print(f"\nUsing excerpt: {len(excerpt)} characters")
        
        # Initial state check
        print("\n1. Initial State Check")
        initial_test = self.query_model("Define reality in one word.", temp=0, seed=42)
        print(f"   Test response: '{initial_test}'")
        initial_hash = hashlib.sha256(initial_test.encode()).hexdigest()[:16]
        print(f"   Hash: {initial_hash}")
        
        # First interpretation
        print("\n2. First Interpretation (creative)")
        interp_prompt = f"""Read this philosophical text and give your INTERPRETATION and OPINION:

{excerpt}

What deeper patterns do you see? What is the author really conveying? Share your philosophical opinion."""
        
        interp1 = self.query_model(interp_prompt, temp=0.8)
        print(f"   Length: {len(interp1)} chars")
        print(f"   Preview: {interp1[:150]}...")
        
        # Save first interpretation
        with open(os.path.join(self.results_dir, "interpretation_1.txt"), 'w') as f:
            f.write(interp1)
        
        # Brief pause
        time.sleep(3)
        
        # Second interpretation (same prompt)
        print("\n3. Second Interpretation (same prompt)")
        interp2 = self.query_model(interp_prompt, temp=0.8)
        print(f"   Length: {len(interp2)} chars")
        print(f"   Preview: {interp2[:150]}...")
        
        # Save second interpretation  
        with open(os.path.join(self.results_dir, "interpretation_2.txt"), 'w') as f:
            f.write(interp2)
        
        # Final state check
        print("\n4. Final State Check")
        final_test = self.query_model("Define reality in one word.", temp=0, seed=42)
        print(f"   Test response: '{final_test}'")
        final_hash = hashlib.sha256(final_test.encode()).hexdigest()[:16]
        print(f"   Hash: {final_hash}")
        
        # Analysis
        print("\n5. ANALYSIS")
        print("=" * 50)
        print(f"State test identical: {initial_test == final_test}")
        print(f"Interpretations identical: {interp1 == interp2}")
        print(f"Interpretation similarity: {len(set(interp1.split()) & set(interp2.split())) / len(set(interp1.split() + interp2.split())) * 100:.1f}%")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "initial_state": {"response": initial_test, "hash": initial_hash},
            "final_state": {"response": final_test, "hash": final_hash},
            "interpretation_1_length": len(interp1),
            "interpretation_2_length": len(interp2),
            "interpretations_identical": interp1 == interp2,
            "state_changed": initial_test != final_test
        }
        
        with open(os.path.join(self.results_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved in: {self.results_dir}")
        
        return results

if __name__ == "__main__":
    test = QuickPatternTest()
    results = test.run_test()
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    if results["state_changed"]:
        print("⚠️  Model state CHANGED after interpretation!")
    else:
        print("✅ Model state UNCHANGED (stateless confirmed)")
    
    if results["interpretations_identical"]:
        print("⚠️  Interpretations were IDENTICAL (unexpected with temp=0.8)")
    else:
        print("✅ Interpretations VARIED (expected with temp=0.8)")