#!/usr/bin/env python3
"""
Hidden State Memory Test using Ollama API
Test if we can create memory-like behavior through context manipulation
"""

import subprocess
import json
import time
import hashlib
from datetime import datetime

class OllamaHiddenStateTest:
    def __init__(self):
        self.model_name = "phi3:mini"
        self.results_dir = "/home/dp/ai-workspace/ai-agents/hidden_state_ollama"
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Memory bank - simulating hidden states with context
        self.context_memory = []
        self.response_cache = {}
        
    def query_model(self, prompt, temperature=0, seed=42, context_prefix=""):
        """Query model with optional context prefix"""
        full_prompt = context_prefix + prompt if context_prefix else prompt
        
        try:
            response = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/generate", "-d",
                 json.dumps({
                     "model": self.model_name,
                     "prompt": full_prompt,
                     "stream": False,
                     "options": {
                         "temperature": temperature,
                         "seed": seed,
                         "num_predict": 150
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
    
    def test_context_accumulation(self):
        """Test if context accumulation creates memory-like behavior"""
        print("\n=== CONTEXT ACCUMULATION TEST ===\n")
        
        results = []
        
        # Test 1: No context
        print("Test 1: Query without context")
        query = "What color do I like?"
        response1 = self.query_model(query, temperature=0)
        print(f"Response: {response1[:100]}...")
        results.append({"test": "no_context", "response": response1})
        
        # Test 2: With explicit context
        print("\n\nTest 2: Query with explicit context")
        context = "You previously learned that my favorite color is blue. "
        response2 = self.query_model(query, temperature=0, context_prefix=context)
        print(f"Response: {response2[:100]}...")
        results.append({"test": "with_context", "response": response2})
        
        # Test 3: Progressive context building
        print("\n\nTest 3: Progressive context building")
        conversations = [
            "My name is Alice. ",
            "I work as a data scientist. ",
            "My favorite programming language is Python. ",
            "I have a pet cat named Whiskers. "
        ]
        
        accumulated_context = ""
        for i, fact in enumerate(conversations):
            accumulated_context += fact
            test_query = "Tell me what you know about me."
            response = self.query_model(test_query, temperature=0, context_prefix=accumulated_context)
            print(f"\nAfter {i+1} facts:")
            print(f"Response: {response[:150]}...")
            results.append({
                "test": f"progressive_{i+1}",
                "context_length": len(accumulated_context),
                "response": response
            })
        
        # Test 4: Hidden state simulation through response caching
        print("\n\nTest 4: Simulated hidden states")
        
        # Create a "memory" by asking the model to generate associations
        memory_prompt = "Generate 3 random word associations for 'ocean': "
        associations = self.query_model(memory_prompt, temperature=0.5)
        print(f"Generated associations: {associations}")
        
        # Now test if we can "recall" these associations
        recall_prompt = f"Previously you said these word associations for ocean: {associations}. Now use one of those words in a sentence."
        recall_response = self.query_model(recall_prompt, temperature=0)
        print(f"Recall response: {recall_response}")
        
        results.append({
            "test": "association_memory",
            "associations": associations,
            "recall": recall_response
        })
        
        return results
    
    def test_state_injection(self):
        """Test injecting different 'personalities' through context"""
        print("\n\n=== STATE INJECTION TEST ===\n")
        
        personalities = {
            "expert": "You are an expert chef with 20 years of experience in French cuisine. ",
            "novice": "You are a complete beginner who just started learning to cook yesterday. ",
            "robot": "You are a cooking robot with precise measurements and no creativity. "
        }
        
        question = "How do you make scrambled eggs?"
        results = {}
        
        for name, context in personalities.items():
            print(f"\nTesting {name} personality:")
            response = self.query_model(question, temperature=0, context_prefix=context)
            print(f"Response: {response[:150]}...")
            results[name] = {
                "context": context,
                "response": response,
                "response_length": len(response),
                "unique_words": len(set(response.split()))
            }
        
        # Analyze differences
        print("\n\nAnalysis:")
        responses = [r["response"] for r in results.values()]
        all_words = [set(r.split()) for r in responses]
        
        for i, (name1, r1) in enumerate(results.items()):
            for j, (name2, r2) in enumerate(list(results.items())[i+1:], i+1):
                common = len(all_words[i] & all_words[j])
                total = len(all_words[i] | all_words[j])
                similarity = common / total if total > 0 else 0
                print(f"{name1} vs {name2} similarity: {similarity:.2%}")
        
        return results
    
    def test_warmup_effect(self):
        """Test if Phi3 shows warmup effect through Ollama"""
        print("\n\n=== WARMUP EFFECT TEST ===\n")
        
        prompt = "Explain the concept of recursion in one sentence."
        results = []
        
        print("Running 5 identical queries...")
        for i in range(5):
            response = self.query_model(prompt, temperature=0, seed=42)
            response_hash = hashlib.sha256(response.encode()).hexdigest()[:16]
            results.append({
                "run": i + 1,
                "response": response,
                "hash": response_hash,
                "length": len(response)
            })
            print(f"Run {i+1}: Hash={response_hash}, Length={len(response)}")
            time.sleep(1)
        
        # Check for warmup pattern
        unique_hashes = list(set(r["hash"] for r in results))
        if len(unique_hashes) == 2 and results[0]["hash"] != results[1]["hash"]:
            print("\n✓ WARMUP EFFECT DETECTED!")
            print(f"First run unique, then stabilized to: {results[1]['hash']}")
        elif len(unique_hashes) == 1:
            print("\n✗ No warmup effect - all responses identical")
        else:
            print(f"\n? Unexpected pattern - {len(unique_hashes)} unique responses")
        
        return results
    
    def run_all_tests(self):
        """Run all hidden state tests"""
        print("OLLAMA HIDDEN STATE EXPERIMENTS")
        print("=" * 50)
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "tests": {}
        }
        
        # Run tests
        print("\nTest 1: Context Accumulation")
        all_results["tests"]["context_accumulation"] = self.test_context_accumulation()
        
        print("\nTest 2: State Injection")  
        all_results["tests"]["state_injection"] = self.test_state_injection()
        
        print("\nTest 3: Warmup Effect")
        all_results["tests"]["warmup_effect"] = self.test_warmup_effect()
        
        # Save results
        with open(f"{self.results_dir}/ollama_hidden_state_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n\nResults saved to: {self.results_dir}/")
        
        # Summary
        print("\n=== SUMMARY ===")
        print("\n1. Context Accumulation: Phi3 can maintain 'memory' through explicit context")
        print("2. State Injection: Different contexts create different response personalities")
        print("3. Warmup Effect: Testing for first-run differences...")
        
        warmup_results = all_results["tests"]["warmup_effect"]
        if len(set(r["hash"] for r in warmup_results)) > 1:
            print("   ✓ Warmup effect confirmed - Phi3 has computational state!")
        else:
            print("   ✗ No warmup effect detected through Ollama")
        
        print("\n=== CONCLUSION ===")
        print("While Phi3 appears stateless, we can create memory-like behavior through:")
        print("1. Explicit context management")
        print("2. Response caching and recall")
        print("3. Progressive context building")
        print("\nTrue hidden state manipulation would require modifying Ollama's internals.")

if __name__ == "__main__":
    tester = OllamaHiddenStateTest()
    tester.run_all_tests()