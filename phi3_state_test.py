#!/usr/bin/env python3
"""
Test Phi3 model state persistence in GPU memory
"""

import subprocess
import json
import time
import hashlib
from datetime import datetime
import torch
import os

class Phi3StateExperiment:
    """Test if Phi3's state changes in GPU memory during use"""
    
    def __init__(self):
        self.model_name = "phi3:mini"
        self.results_dir = "/home/dp/ai-workspace/ai-agents/phi3_state_test"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def capture_model_state(self, label):
        """Capture current model state through various methods"""
        print(f"\n📸 Capturing model state: {label}")
        state = {
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'gpu_memory_mb': self._get_gpu_memory(),
            'model_info': self._get_model_info(),
            'response_samples': self._get_response_samples()
        }
        
        # Save state
        filename = f"state_{label}_{datetime.now().strftime('%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"  ✓ State saved: {filename}")
        print(f"  GPU Memory: {state['gpu_memory_mb']:.1f} MB")
        
        return state
    
    def _get_gpu_memory(self):
        """Get current GPU memory allocation"""
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
        return 0
    
    def _get_model_info(self):
        """Get model information from Ollama"""
        try:
            response = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/show", "-d",
                 json.dumps({"name": self.model_name})],
                capture_output=True, text=True
            )
            if response.returncode == 0:
                return json.loads(response.stdout)
        except:
            pass
        return {}
    
    def _get_response_samples(self):
        """Get deterministic responses to test model state"""
        test_prompts = [
            {"prompt": "Complete: 2+2=", "temp": 0},
            {"prompt": "The capital of France is", "temp": 0},
            {"prompt": "Define consciousness in one word:", "temp": 0}
        ]
        
        samples = []
        for test in test_prompts:
            response = self._query_model(test["prompt"], temperature=test["temp"])
            samples.append({
                'prompt': test["prompt"],
                'response': response,
                'hash': hashlib.md5(response.encode()).hexdigest()
            })
        
        return samples
    
    def _query_model(self, prompt, temperature=0):
        """Query the model with specific parameters"""
        try:
            response = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/generate", "-d",
                 json.dumps({
                     "model": self.model_name,
                     "prompt": prompt,
                     "temperature": temperature,
                     "stream": False,
                     "options": {
                         "seed": 42,  # Fixed seed for determinism
                         "top_k": 1   # Greedy decoding
                     }
                 })],
                capture_output=True, text=True, timeout=15
            )
            
            if response.returncode == 0:
                result = json.loads(response.stdout)
                return result.get('response', '').strip()
        except:
            pass
        return ""
    
    def run_intensive_workload(self):
        """Run intensive workload to potentially change model state"""
        print("\n🔥 Running intensive workload...")
        
        workload_results = {
            'start_time': datetime.now().isoformat(),
            'tasks': []
        }
        
        # Task 1: Large context processing
        print("  Task 1: Large context processing")
        large_context = " ".join([f"Item {i}: " + "x"*50 for i in range(50)])
        large_prompt = f"{large_context}\n\nSummarize the above in 10 words."
        
        start = time.time()
        response = self._query_model(large_prompt, temperature=0.8)
        duration = time.time() - start
        
        workload_results['tasks'].append({
            'name': 'large_context',
            'duration': duration,
            'input_size': len(large_prompt),
            'output_size': len(response)
        })
        print(f"    Duration: {duration:.2f}s")
        
        # Task 2: Rapid-fire queries
        print("  Task 2: Rapid-fire queries (20 requests)")
        start = time.time()
        for i in range(20):
            self._query_model(f"Count to {i}", temperature=0)
            print(f"    Request {i+1}/20", end='\r')
        duration = time.time() - start
        print(f"\n    Duration: {duration:.2f}s")
        
        workload_results['tasks'].append({
            'name': 'rapid_fire',
            'duration': duration,
            'count': 20
        })
        
        # Task 3: High temperature generation
        print("  Task 3: High temperature creative generation")
        creative_prompts = [
            "Invent a new color and describe it:",
            "Create a word that doesn't exist:",
            "Imagine a emotion humans don't have:"
        ]
        
        for i, prompt in enumerate(creative_prompts):
            response = self._query_model(prompt, temperature=1.5)
            workload_results['tasks'].append({
                'name': f'creative_{i}',
                'prompt': prompt,
                'response_length': len(response)
            })
        
        # Task 4: Adversarial prompts
        print("  Task 4: Edge case processing")
        edge_cases = [
            "�����",  # Unicode stress
            "a" * 1000,  # Repetition
            "Explain: " * 100,  # Recursive
        ]
        
        for i, prompt in enumerate(edge_cases):
            try:
                response = self._query_model(prompt[:500], temperature=0.5)
                workload_results['tasks'].append({
                    'name': f'edge_case_{i}',
                    'handled': True
                })
            except:
                workload_results['tasks'].append({
                    'name': f'edge_case_{i}',
                    'handled': False
                })
        
        workload_results['end_time'] = datetime.now().isoformat()
        
        # Save workload results
        workload_path = os.path.join(self.results_dir, "workload_results.json")
        with open(workload_path, 'w') as f:
            json.dump(workload_results, f, indent=2)
        
        print("\n  ✓ Workload complete")
        return workload_results
    
    def analyze_state_changes(self, states):
        """Analyze differences between captured states"""
        print("\n📊 Analyzing state changes...")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'states_compared': len(states),
            'findings': {}
        }
        
        # Compare GPU memory
        memory_values = [s['gpu_memory_mb'] for s in states]
        analysis['findings']['gpu_memory'] = {
            'values': memory_values,
            'changed': max(memory_values) - min(memory_values) > 1,
            'delta': max(memory_values) - min(memory_values)
        }
        
        # Compare response determinism
        response_hashes = []
        for state in states:
            hashes = [s['hash'] for s in state['response_samples']]
            response_hashes.append(hashes)
        
        # Check if responses are identical across states
        deterministic = all(response_hashes[0] == h for h in response_hashes[1:])
        analysis['findings']['determinism'] = {
            'is_deterministic': deterministic,
            'response_hashes': response_hashes
        }
        
        # Display analysis
        print(f"\n  GPU Memory:")
        print(f"    Range: {min(memory_values):.1f} - {max(memory_values):.1f} MB")
        print(f"    Delta: {analysis['findings']['gpu_memory']['delta']:.1f} MB")
        print(f"    Changed: {'YES' if analysis['findings']['gpu_memory']['changed'] else 'NO'}")
        
        print(f"\n  Response Determinism:")
        print(f"    Deterministic: {'YES' if deterministic else 'NO'}")
        
        if not deterministic:
            print("\n  Response variations detected:")
            for i, prompt_data in enumerate(states[0]['response_samples']):
                prompt = prompt_data['prompt']
                print(f"\n    '{prompt}'")
                for j, state in enumerate(states):
                    response = state['response_samples'][i]['response'][:50]
                    print(f"      State {j}: {response}...")
        
        # Save analysis
        analysis_path = os.path.join(self.results_dir, "state_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis


def main():
    """Run the complete experiment"""
    print("=" * 60)
    print("PHI3 GPU STATE PERSISTENCE EXPERIMENT")
    print("=" * 60)
    
    experiment = Phi3StateExperiment()
    states = []
    
    # Capture initial state
    print("\nPhase 1: Initial State")
    states.append(experiment.capture_model_state("initial"))
    time.sleep(2)
    
    # Run intensive workload
    print("\nPhase 2: Intensive Workload")
    experiment.run_intensive_workload()
    time.sleep(2)
    
    # Capture post-workload state
    print("\nPhase 3: Post-Workload State")
    states.append(experiment.capture_model_state("post_workload"))
    time.sleep(2)
    
    # Let model "cool down"
    print("\nPhase 4: Cooldown (30 seconds)")
    print("  Waiting for potential state settling...")
    time.sleep(30)
    
    # Capture final state
    print("\nPhase 5: Final State")
    states.append(experiment.capture_model_state("final"))
    
    # Analyze results
    print("\nPhase 6: Analysis")
    analysis = experiment.analyze_state_changes(states)
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"✓ States captured: {len(states)}")
    print(f"✓ Workload tasks: Multiple intensive operations")
    print(f"✓ GPU memory stable: {'YES' if not analysis['findings']['gpu_memory']['changed'] else 'NO'}")
    print(f"✓ Responses deterministic: {'YES' if analysis['findings']['determinism']['is_deterministic'] else 'NO'}")
    print(f"\nResults saved in: {experiment.results_dir}")
    
    # Key finding
    if analysis['findings']['determinism']['is_deterministic'] and not analysis['findings']['gpu_memory']['changed']:
        print("\n🔬 FINDING: Phi3 model state in GPU appears to be STATELESS")
        print("   The model produces identical outputs regardless of workload history.")
    else:
        print("\n🔬 FINDING: Phi3 model shows signs of STATEFUL behavior")
        print("   Either memory or outputs changed after intensive use.")


if __name__ == "__main__":
    main()