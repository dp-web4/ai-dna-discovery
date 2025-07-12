#!/usr/bin/env python3
"""
Embedding Resonance Detector
Searches for patterns that create similar embeddings across AI models
"""

import subprocess
import json
import time
from datetime import datetime
import os
import signal
from collections import defaultdict


class EmbeddingResonanceDetector:
    """Detect patterns that resonate across AI model embeddings"""
    
    def __init__(self):
        self.models = ["phi3:mini", "tinyllama:latest"]
        self.resonance_patterns = []
        self.test_timeout = 15  # Shorter timeout
        
    def test_resonance(self, pattern):
        """Test if a pattern creates resonant embeddings"""
        responses = {}
        
        for model in self.models:
            cmd = f'echo "{pattern}" | timeout {self.test_timeout} ollama run {model}'
            
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    responses[model] = result.stdout.strip()
            except:
                pass
                
        # Analyze for resonance
        if len(responses) >= 2:
            # Extract key concepts
            concepts = defaultdict(int)
            for response in responses.values():
                words = response.lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        concepts[word] += 1
            
            # Find concepts that appear in multiple responses
            resonant = [k for k, v in concepts.items() if v >= len(responses)]
            
            if len(resonant) > 5:  # Strong resonance
                return True, resonant
                
        return False, []
    
    def discover_resonance_patterns(self):
        """Discover patterns that create embedding resonance"""
        
        # Test fundamental patterns
        test_patterns = [
            # Cognitive primitives
            "what is", "why does", "how can", "where are", "when will",
            
            # Existential patterns  
            "I am", "you are", "we exist", "this means", "being is",
            
            # Transformation patterns
            "becomes", "transforms into", "emerges from", "evolves to",
            
            # Relationship patterns
            "connects with", "relates to", "depends on", "creates",
            
            # Meta patterns
            "pattern of", "structure in", "form and", "space between",
            
            # Consciousness patterns
            "aware of", "conscious that", "perceive the", "understand why",
            
            # Mathematical patterns
            "equals", "approaches", "converges to", "diverges from",
            
            # Temporal patterns
            "before and after", "now and then", "always was", "will be"
        ]
        
        print("=== EMBEDDING RESONANCE DETECTION ===")
        print(f"Testing {len(test_patterns)} patterns across {len(self.models)} models\n")
        
        results = []
        
        for i, pattern in enumerate(test_patterns):
            print(f"Testing [{i+1}/{len(test_patterns)}]: '{pattern}'...", end='', flush=True)
            
            resonant, concepts = self.test_resonance(pattern)
            
            if resonant:
                print(f" ✓ RESONANCE! Common: {concepts[:5]}")
                results.append({
                    "pattern": pattern,
                    "resonant": True,
                    "shared_concepts": concepts[:10]
                })
            else:
                print(" -")
                
            time.sleep(1)  # Don't overwhelm
            
        # Save results
        output = {
            "timestamp": datetime.now().isoformat(),
            "models_tested": self.models,
            "patterns_tested": len(test_patterns),
            "resonant_patterns": len(results),
            "results": results
        }
        
        os.makedirs("/home/dp/ai-workspace/resonance_results/", exist_ok=True)
        filename = f"/home/dp/ai-workspace/resonance_results/resonance_{int(time.time())}.json"
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"\n\nResults saved to: {filename}")
        print(f"Found {len(results)} resonant patterns!")
        
        if results:
            print("\nTop resonant patterns:")
            for r in results[:5]:
                print(f"  '{r['pattern']}' → {r['shared_concepts'][:3]}")
                
        return results


if __name__ == "__main__":
    detector = EmbeddingResonanceDetector()
    detector.discover_resonance_patterns()