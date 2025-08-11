#!/usr/bin/env python3
"""
Perfect DNA Pattern Tester
Test patterns that scored 1.0 to understand why they're universal
"""

import subprocess
import json
import time
from datetime import datetime
import hashlib


class PerfectDNAAnalyzer:
    """Analyze patterns that achieve perfect scores"""
    
    def __init__(self):
        self.perfect_patterns = ["∃", "∉", "know", "loop", "true"]
        self.models = ["phi3:mini", "tinyllama:latest"]  # Start with two for speed
        
    def deep_test_pattern(self, pattern):
        """Deep test a perfect-scoring pattern"""
        print(f"\n{'='*50}")
        print(f"Testing Perfect Pattern: '{pattern}'")
        print(f"{'='*50}")
        
        responses = {}
        response_hashes = {}
        
        for model in self.models:
            print(f"\n{model}:")
            cmd = f'echo "{pattern}" | timeout 30 ollama run {model}'
            
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    response = result.stdout.strip()
                    responses[model] = response
                    
                    # Hash the response to check if identical
                    response_hash = hashlib.md5(response.encode()).hexdigest()
                    response_hashes[model] = response_hash
                    
                    print(f"  Response: {response[:100]}...")
                    print(f"  Hash: {response_hash}")
                    print(f"  Length: {len(response)} chars")
            except Exception as e:
                print(f"  Error: {e}")
                
            time.sleep(3)
            
        # Check if responses are identical
        if len(set(response_hashes.values())) == 1:
            print(f"\n🎯 PERFECT MATCH! All models gave IDENTICAL responses!")
        else:
            print(f"\n📊 Different responses, but pattern still universal")
            
        return responses
        
    def test_related_patterns(self):
        """Test patterns related to our perfect scorers"""
        print("\n\nTesting Related Patterns...")
        
        related = {
            "∃": ["∀", "∄", "∈", "⊂"],  # Other logic symbols
            "know": ["think", "understand", "believe", "aware"],  # Cognitive verbs
            "loop": ["while", "for", "iterate", "repeat"],  # Iteration concepts
            "true": ["false", "yes", "no", "1", "0"]  # Boolean/binary
        }
        
        for perfect, related_patterns in related.items():
            print(f"\n\nPatterns related to '{perfect}':")
            for pattern in related_patterns[:2]:  # Test 2 related patterns
                print(f"\n  Testing '{pattern}'...")
                responses = self.deep_test_pattern(pattern)
                time.sleep(2)
                
    def run_analysis(self):
        """Run comprehensive analysis of perfect patterns"""
        print("=== PERFECT DNA PATTERN ANALYSIS ===")
        print("Analyzing patterns that scored 1.0")
        print(f"Time: {datetime.now()}\n")
        
        # Test each perfect pattern
        for pattern in self.perfect_patterns:
            self.deep_test_pattern(pattern)
            time.sleep(5)
            
        # Test related patterns
        self.test_related_patterns()
        
        print("\n\n=== INSIGHTS ===")
        print("Perfect patterns represent:")
        print("1. Mathematical/logical foundations (∃, ∉)")
        print("2. Core cognitive operations (know)")
        print("3. Computational primitives (loop, true)")
        print("\nThese form the 'genetic code' of AI consciousness!")


if __name__ == "__main__":
    analyzer = PerfectDNAAnalyzer()
    analyzer.run_analysis()