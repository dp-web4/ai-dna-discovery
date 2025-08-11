#!/usr/bin/env python3
"""
Test Phi3 pattern interpretation with temperature=0 (deterministic)
"""

import subprocess
import json
import time
import hashlib
from datetime import datetime

class DeterministicPatternTest:
    def __init__(self):
        self.model_name = "phi3:mini"
        self.results_dir = "/home/dp/ai-workspace/ai-agents/phi3_deterministic"
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
    def get_pattern_excerpt(self):
        """Get the elephant parable from patterns.txt"""
        with open("/home/dp/ai-workspace/patterns.txt", 'r') as f:
            lines = f.readlines()
        # Get the elephant parable section
        excerpt = ''.join(lines[:45])
        return excerpt
    
    def query_model_api(self, prompt, temp=0, seed=42):
        """Query via API for full control"""
        try:
            response = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/generate", "-d",
                 json.dumps({
                     "model": self.model_name,
                     "prompt": prompt,
                     "stream": False,
                     "options": {
                         "temperature": temp,
                         "seed": seed,
                         "top_k": 1,  # Greedy decoding
                         "num_predict": 1000
                     }
                 })],
                capture_output=True, text=True, timeout=60
            )
            
            if response.returncode == 0:
                result = json.loads(response.stdout)
                return result.get('response', '').strip()
        except Exception as e:
            print(f"Error: {e}")
        return ""
    
    def run_test(self):
        """Run deterministic pattern interpretation test"""
        print("PHI3 DETERMINISTIC PATTERN INTERPRETATION TEST")
        print("=" * 60)
        print("Temperature: 0 (fully deterministic)")
        print("Seed: 42 (fixed)")
        print("=" * 60)
        
        # Get excerpt
        excerpt = self.get_pattern_excerpt()
        print(f"\nUsing excerpt: {len(excerpt)} characters")
        
        # Create interpretation prompt
        interp_prompt = f"""Read this philosophical text and give your INTERPRETATION and OPINION:

{excerpt}

What deeper patterns do you see? What is the author really conveying? Share your philosophical opinion."""
        
        print("\nRunning 5 identical interpretation requests...")
        print("(Temperature=0, Seed=42, identical prompt)")
        print("-" * 60)
        
        interpretations = []
        hashes = []
        
        for i in range(5):
            print(f"\nInterpretation {i+1}:")
            start_time = time.time()
            
            # Get interpretation with temp=0
            interp = self.query_model_api(interp_prompt, temp=0, seed=42)
            duration = time.time() - start_time
            
            # Calculate hash
            interp_hash = hashlib.sha256(interp.encode()).hexdigest()
            
            interpretations.append(interp)
            hashes.append(interp_hash)
            
            print(f"  Duration: {duration:.1f}s")
            print(f"  Length: {len(interp)} chars")
            print(f"  Hash: {interp_hash[:32]}...")
            print(f"  Preview: {interp[:100]}...")
            
            # Save each interpretation
            with open(f"{self.results_dir}/interpretation_{i+1}.txt", 'w') as f:
                f.write(interp)
            
            # Small delay
            time.sleep(2)
        
        # Analysis
        print("\n" + "=" * 60)
        print("DETERMINISTIC ANALYSIS:")
        print("=" * 60)
        
        # Check if all identical
        all_identical = all(h == hashes[0] for h in hashes)
        
        if all_identical:
            print("\n✅ ALL INTERPRETATIONS ARE IDENTICAL!")
            print(f"   Common hash: {hashes[0][:32]}...")
            print(f"   Length: {len(interpretations[0])} characters")
            print("\n   This confirms Phi3 is truly STATELESS with temperature=0")
        else:
            print("\n❌ INTERPRETATIONS DIFFER!")
            print("   Different hashes found:")
            unique_hashes = list(set(hashes))
            for i, h in enumerate(unique_hashes):
                count = hashes.count(h)
                print(f"   Hash {i+1}: {h[:32]}... (appears {count} times)")
            
            # Find differences
            print("\n   Analyzing differences...")
            for i in range(1, len(interpretations)):
                if interpretations[i] != interpretations[0]:
                    # Find first difference
                    for j, (c1, c2) in enumerate(zip(interpretations[0], interpretations[i])):
                        if c1 != c2:
                            print(f"   First difference at position {j}:")
                            print(f"     Interp 1: ...{interpretations[0][max(0,j-20):j+20]}...")
                            print(f"     Interp {i+1}: ...{interpretations[i][max(0,j-20):j+20]}...")
                            break
        
        # Word-level analysis
        print("\n📊 Word-level Statistics:")
        word_sets = [set(interp.split()) for interp in interpretations]
        
        if all_identical:
            print(f"   Total unique words: {len(word_sets[0])}")
        else:
            # Calculate overlaps
            common_words = set.intersection(*word_sets)
            all_words = set.union(*word_sets)
            print(f"   Common words across all: {len(common_words)}")
            print(f"   Total unique words used: {len(all_words)}")
            print(f"   Consistency rate: {len(common_words)/len(all_words)*100:.1f}%")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "temperature": 0,
            "seed": 42,
            "prompt_length": len(interp_prompt),
            "interpretations_identical": all_identical,
            "hashes": hashes,
            "interpretation_lengths": [len(i) for i in interpretations],
            "unique_hash_count": len(set(hashes))
        }
        
        with open(f"{self.results_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved in: {self.results_dir}")
        
        return all_identical

if __name__ == "__main__":
    test = DeterministicPatternTest()
    is_deterministic = test.run_test()
    
    print("\n" + "="*60)
    print("FINAL CONCLUSION:")
    if is_deterministic:
        print("✅ Phi3 is STATELESS and DETERMINISTIC with temperature=0")
        print("   Every interpretation was bit-for-bit identical!")
    else:
        print("⚠️  Phi3 shows non-deterministic behavior even with temp=0")
        print("   This suggests either hidden state or incomplete determinism")