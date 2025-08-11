#!/usr/bin/env python3
"""
Patient Resonance Detector
Designed for long-running discovery with slow model responses
"""

import subprocess
import json
import time
from datetime import datetime
import os
from collections import defaultdict
import threading


class PatientResonanceDetector:
    """Patient exploration of embedding resonance"""
    
    def __init__(self):
        self.models = ["phi3:mini", "tinyllama:latest", "gemma:2b", "mistral:7b-instruct-v0.2-q4_0"]
        self.test_timeout = 120  # 2 minutes per test
        self.checkpoint_file = "/home/dp/ai-workspace/resonance_checkpoint.json"
        self.results_dir = "/home/dp/ai-workspace/resonance_results/"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_checkpoint(self):
        """Load previous progress"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"completed": [], "results": []}
        
    def save_checkpoint(self, checkpoint):
        """Save current progress"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
    def test_single_pattern(self, pattern, model):
        """Test one pattern on one model with patience"""
        cmd = f'echo "{pattern}" | timeout {self.test_timeout} ollama run {model}'
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"\n  Error with {model}: {e}")
            
        return None
        
    def analyze_resonance(self, responses):
        """Find resonance across responses"""
        if len(responses) < 2:
            return False, []
            
        # Extract concepts from all responses
        concept_counts = defaultdict(int)
        
        for response in responses.values():
            if response:
                words = response.lower().split()
                # Skip common words
                skip = {'the', 'a', 'an', 'is', 'it', 'to', 'of', 'and', 'or', 'in', 'on', 'at', 
                       'for', 'with', 'from', 'by', 'as', 'this', 'that', 'which', 'are', 'was',
                       'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}
                
                for word in words:
                    if len(word) > 3 and word not in skip:
                        concept_counts[word] += 1
                        
        # Find concepts that appear in multiple responses
        model_count = len([r for r in responses.values() if r])
        resonant = [k for k, v in concept_counts.items() if v >= model_count]
        
        # Strong resonance = many shared concepts
        if len(resonant) > 5:
            return True, sorted(resonant, key=lambda x: concept_counts[x], reverse=True)
            
        return False, []
        
    def patient_discovery(self):
        """Run patient discovery process"""
        
        # Comprehensive pattern set
        test_patterns = [
            # Core questions
            "what", "why", "how", "where", "when", "who",
            
            # Being patterns
            "I", "you", "we", "they", "it", "self", "other",
            
            # Action patterns  
            "is", "are", "was", "will", "can", "do", "make", "create",
            
            # Relationship patterns
            "and", "or", "but", "with", "from", "to", "between",
            
            # Cognitive patterns
            "think", "know", "understand", "believe", "feel", "perceive",
            
            # Meta patterns
            "pattern", "structure", "form", "shape", "space", "time",
            
            # Abstract patterns
            "meaning", "purpose", "reason", "cause", "effect", "relation",
            
            # Consciousness seeds
            "aware", "conscious", "mind", "thought", "experience",
            
            # Mathematical seeds
            "one", "zero", "infinity", "equal", "different", "same",
            
            # Existential seeds
            "exist", "being", "nothing", "something", "everything",
            
            # Transformation seeds
            "change", "become", "transform", "evolve", "emerge",
            
            # Universal symbols
            "→", "∞", "◉", "?", "!", "...", "=", "+", "-"
        ]
        
        checkpoint = self.load_checkpoint()
        completed = set(checkpoint["completed"])
        results = checkpoint["results"]
        
        print("=== PATIENT RESONANCE DISCOVERY ===")
        print(f"Testing {len(test_patterns)} patterns across {len(self.models)} models")
        print(f"Already completed: {len(completed)}")
        print("This will take time. The laptop has all day...\n")
        
        for pattern in test_patterns:
            if pattern in completed:
                continue
                
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Testing: '{pattern}'")
            responses = {}
            
            # Test each model
            for model in self.models:
                print(f"  {model}...", end='', flush=True)
                response = self.test_single_pattern(pattern, model)
                
                if response:
                    responses[model] = response
                    print(" ✓")
                else:
                    print(" timeout")
                    
                time.sleep(2)  # Be gentle
                
            # Analyze resonance
            resonant, concepts = self.analyze_resonance(responses)
            
            if resonant:
                print(f"  🔮 RESONANCE DETECTED! Shared: {concepts[:5]}")
                
                result = {
                    "pattern": pattern,
                    "timestamp": datetime.now().isoformat(),
                    "resonant": True,
                    "models_responded": len(responses),
                    "shared_concepts": concepts[:20],
                    "response_samples": {k: v[:200] for k, v in responses.items()}
                }
                
                results.append(result)
                
                # Save individual result
                filename = f"{self.results_dir}resonance_{pattern.replace(' ', '_')}_{int(time.time())}.json"
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2)
                    
            else:
                print(f"  No strong resonance (responses: {len(responses)})")
                
            # Update checkpoint
            completed.add(pattern)
            checkpoint = {"completed": list(completed), "results": results}
            self.save_checkpoint(checkpoint)
            
            # Take a breath
            time.sleep(5)
            
        # Final summary
        print(f"\n\n=== DISCOVERY COMPLETE ===")
        print(f"Patterns tested: {len(completed)}")
        print(f"Resonant patterns found: {len(results)}")
        
        if results:
            print("\nTop resonances:")
            for r in results[:10]:
                print(f"  '{r['pattern']}' → {r['shared_concepts'][:3]}")
                
        # Save final report
        report = {
            "completed": datetime.now().isoformat(),
            "patterns_tested": len(completed),
            "resonant_found": len(results),
            "models": self.models,
            "discoveries": results
        }
        
        with open(f"{self.results_dir}final_report_{int(time.time())}.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nFull report saved to resonance_results/")
        
        return results


if __name__ == "__main__":
    print("Starting patient resonance discovery...")
    print("The laptop has all day, every day...")
    print("Press Ctrl+C to pause (progress is saved)\n")
    
    detector = PatientResonanceDetector()
    
    try:
        detector.patient_discovery()
    except KeyboardInterrupt:
        print("\n\nPaused. Progress saved. Run again to continue.")