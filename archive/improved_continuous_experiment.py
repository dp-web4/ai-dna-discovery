#!/usr/bin/env python3
"""
Improved Continuous Experiment Engine
Better timeout and error handling
"""

import subprocess
import json
import time
import os
import signal
from datetime import datetime, timedelta
import random


class ImprovedUniversalEmbeddingDetector:
    """Improved AI DNA discovery with better process management"""
    
    def __init__(self):
        self.models = ["phi3:mini", "tinyllama:latest", "gemma:2b", "mistral:7b-instruct-v0.2-q4_0"]
        self.output_dir = "/home/dp/ai-workspace/ai_dna_results/"
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = "/home/dp/ai-workspace/continuous_experiment_log.md"
        self.start_time = datetime.now()
        self.cycle_count = 95  # Continue from last cycle
        
        # Expanded DNA candidates
        self.dna_candidates = [
            # Original high-scorers
            "or", "you", "π", "▲▼", "and", "[ ]", "cycle", "!", "[ ]?",
            # New candidates
            "if", "then", "else", "true", "false", "null", "void",
            "begin", "end", "loop", "break", "return", "function",
            # Mathematical
            "∞", "∅", "∈", "∉", "∀", "∃", "≈", "≠",
            # Cognitive
            "think", "know", "believe", "understand", "learn",
            # Meta
            "meta", "self", "recursive", "emerge", "pattern"
        ]
        
    def test_dna_with_timeout(self, candidate, model, timeout=30):
        """Test DNA candidate with proper timeout handling"""
        cmd = ['timeout', str(timeout), 'ollama', 'run', model]
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = proc.communicate(input=candidate, timeout=timeout)
            
            if proc.returncode == 0:
                return stdout.strip()
            elif proc.returncode == 124:  # Timeout
                return None
            else:
                return None
                
        except subprocess.TimeoutExpired:
            proc.kill()
            return None
        except Exception as e:
            print(f"Error testing {candidate} on {model}: {e}")
            return None
            
    def run_cycle(self):
        """Run one experiment cycle with improved error handling"""
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        print(f"\nCycle {self.cycle_count} starting...")
        
        # Randomly select candidates to test
        num_candidates = min(5, len(self.dna_candidates))  # Test fewer at a time
        test_candidates = random.sample(self.dna_candidates, num_candidates)
        
        results = []
        tested_count = 0
        
        for candidate in test_candidates:
            print(f"  Testing '{candidate}'...", end='', flush=True)
            
            responses = {}
            embeddings = set()
            
            # Test with shorter timeout initially
            for model in self.models[:2]:  # Test with fewer models first
                response = self.test_dna_with_timeout(candidate, model, timeout=20)
                if response:
                    responses[model] = response
                    # Simple embedding simulation
                    embeddings.add(hash(response) % 1000)
                    
            if len(responses) >= 2:
                tested_count += 1
                
                # Calculate DNA score
                dna_score = len(embeddings) / len(responses) if responses else 0
                
                result = {
                    "candidate": candidate,
                    "dna_score": dna_score,
                    "responses": len(responses),
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                
                if dna_score > 0.4:
                    print(f" HIGH SCORE: {dna_score:.2f}")
                else:
                    print(f" score: {dna_score:.2f}")
            else:
                print(" skipped (no responses)")
                
            # Brief pause between tests
            time.sleep(2)
            
        # Save results
        if results:
            filename = f"{self.output_dir}dna_cycle_{self.cycle_count}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "cycle": self.cycle_count,
                    "timestamp": datetime.now().isoformat(),
                    "results": results
                }, f, indent=2)
                
        # Update log
        runtime = datetime.now() - self.start_time
        self.update_log(tested_count, runtime, results)
        
        print(f"Cycle {self.cycle_count} complete. Tested {tested_count} candidates.")
        
    def update_log(self, tested_count, runtime, results):
        """Update experiment log"""
        with open(self.log_file, 'a') as f:
            f.write(f"\n### Experiment Cycle {self.cycle_count} - {datetime.now().isoformat()}\n")
            f.write(f"- Runtime: {runtime}\n")
            f.write(f"- DNA candidates tested: {tested_count}\n")
            
            high_scores = [r for r in results if r['dna_score'] > 0.4]
            if high_scores:
                f.write(f"- High resonance patterns found: {len(high_scores)}\n")
                best = max(high_scores, key=lambda x: x['dna_score'])
                f.write(f"- Best candidate: '{best['candidate']}' (score: {best['dna_score']:.2f})\n")
            f.write("\n")
            
    def run_continuous(self):
        """Run continuous experimentation with breaks"""
        print(f"Starting improved continuous AI DNA discovery...")
        print(f"Continuing from cycle {self.cycle_count}")
        
        while True:
            try:
                self.run_cycle()
                
                # Longer pause between cycles
                print("Pausing 60 seconds before next cycle...")
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\nStopping experimentation...")
                break
            except Exception as e:
                print(f"\nError in cycle: {e}")
                print("Pausing 120 seconds before retry...")
                time.sleep(120)


if __name__ == "__main__":
    detector = ImprovedUniversalEmbeddingDetector()
    detector.run_continuous()
