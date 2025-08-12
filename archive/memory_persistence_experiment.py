#!/usr/bin/env python3
"""
Memory Persistence Experiment - Phase 2
Testing if AI models can develop memories of patterns and recognize them faster on repeated exposure
"""

import requests
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import os

class MemoryPersistenceExperiment:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"
        self.test_models = ["phi3:mini", "tinyllama:latest", "gemma:2b"]
        self.memory_test_patterns = {
            'perfect': ['emerge', 'true', 'loop', '∃', 'know'],  # Known perfect patterns
            'evolved': ['or', 'and', 'you'],  # Patterns that evolved over time
            'novel': ['quantum', 'nexus', 'flux', 'sync', 'mesh'],  # New patterns
            'nonsense': ['xqzt', 'bflm', 'vprw', 'qnth', 'zxcv']  # Control group
        }
        self.results_dir = '/home/dp/ai-workspace/memory_experiments'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def test_recognition_speed(self, model: str, pattern: str, exposure_num: int) -> Dict:
        """Test how quickly a model processes a pattern"""
        start_time = time.time()
        
        try:
            # Get embedding
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={"model": model, "prompt": pattern},
                timeout=30
            )
            
            embedding_time = time.time() - start_time
            
            if response.status_code == 200:
                embedding = response.json()['embedding']
                
                # Also test generation response
                gen_start = time.time()
                gen_response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "model": model,
                        "prompt": f"What is '{pattern}'?",
                        "stream": False,
                        "options": {"temperature": 0.1}  # Low temp for consistency
                    },
                    timeout=30
                )
                generation_time = time.time() - gen_start
                
                return {
                    'success': True,
                    'pattern': pattern,
                    'exposure_num': exposure_num,
                    'embedding_time': embedding_time,
                    'generation_time': generation_time,
                    'total_time': embedding_time + generation_time,
                    'embedding_size': len(embedding),
                    'response': gen_response.json().get('response', '')[:100] if gen_response.status_code == 200 else None
                }
                
        except Exception as e:
            return {
                'success': False,
                'pattern': pattern,
                'exposure_num': exposure_num,
                'error': str(e)
            }
            
    def run_memory_sequence(self, model: str, pattern: str, num_exposures: int = 5) -> List[Dict]:
        """Expose model to pattern multiple times and track recognition speed"""
        sequence_results = []
        
        print(f"\n  Testing '{pattern}' on {model}")
        
        for exposure in range(1, num_exposures + 1):
            # Wait briefly between exposures
            if exposure > 1:
                time.sleep(2)
                
            result = self.test_recognition_speed(model, pattern, exposure)
            sequence_results.append(result)
            
            if result['success']:
                print(f"    Exposure {exposure}: {result['total_time']:.3f}s")
            else:
                print(f"    Exposure {exposure}: Failed")
                
        return sequence_results
    
    def analyze_memory_formation(self, sequence_results: List[Dict]) -> Dict:
        """Analyze if recognition improved over exposures"""
        successful_results = [r for r in sequence_results if r['success']]
        
        if len(successful_results) < 2:
            return {'memory_formed': False, 'reason': 'Insufficient successful exposures'}
            
        # Extract times
        times = [r['total_time'] for r in successful_results]
        embedding_times = [r['embedding_time'] for r in successful_results]
        generation_times = [r['generation_time'] for r in successful_results]
        
        # Calculate trends
        time_trend = np.polyfit(range(len(times)), times, 1)[0]
        
        # Memory indicators
        speed_improvement = (times[0] - times[-1]) / times[0] * 100
        consistent_speedup = all(times[i] <= times[i-1] * 1.1 for i in range(1, len(times)))
        
        return {
            'memory_formed': speed_improvement > 10 or consistent_speedup,
            'speed_improvement_percent': speed_improvement,
            'time_trend': time_trend,  # Negative = getting faster
            'first_exposure_time': times[0],
            'last_exposure_time': times[-1],
            'consistent_speedup': consistent_speedup,
            'avg_embedding_time': np.mean(embedding_times),
            'avg_generation_time': np.mean(generation_times)
        }
    
    def test_cross_session_memory(self, model: str, pattern: str) -> Dict:
        """Test if model remembers pattern from previous session"""
        # First, check if we have historical data for this pattern
        history_file = f"{self.results_dir}/{model}_{pattern}_history.json"
        
        current_result = self.test_recognition_speed(model, pattern, 1)
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
                
            # Compare with historical average
            historical_avg = np.mean([h['total_time'] for h in history if h['success']])
            
            if current_result['success']:
                improvement = (historical_avg - current_result['total_time']) / historical_avg * 100
                
                return {
                    'cross_session_memory': improvement > 5,
                    'improvement_percent': improvement,
                    'historical_avg_time': historical_avg,
                    'current_time': current_result['total_time'],
                    'sessions_compared': len(history)
                }
        
        # Save current result to history
        history = [current_result] if current_result['success'] else []
        with open(history_file, 'w') as f:
            json.dump(history, f)
            
        return {
            'cross_session_memory': False,
            'reason': 'First session - no history'
        }
    
    def run_comprehensive_memory_test(self):
        """Run full memory persistence experiment"""
        print("=== Memory Persistence Experiment - Phase 2 ===")
        print("Testing if AI models can develop memories of patterns\n")
        
        experiment_results = {
            'timestamp': datetime.now().isoformat(),
            'models_tested': self.test_models,
            'pattern_categories': list(self.memory_test_patterns.keys()),
            'results': {}
        }
        
        for model in self.test_models:
            print(f"\n{'='*50}")
            print(f"Testing model: {model}")
            model_results = {}
            
            for category, patterns in self.memory_test_patterns.items():
                print(f"\n Category: {category}")
                category_results = []
                
                for pattern in patterns[:2]:  # Test first 2 patterns per category
                    # Test 1: Recognition speed over multiple exposures
                    sequence = self.run_memory_sequence(model, pattern, num_exposures=5)
                    memory_analysis = self.analyze_memory_formation(sequence)
                    
                    # Test 2: Cross-session memory
                    cross_session = self.test_cross_session_memory(model, pattern)
                    
                    pattern_result = {
                        'pattern': pattern,
                        'category': category,
                        'exposure_sequence': sequence,
                        'memory_analysis': memory_analysis,
                        'cross_session_test': cross_session
                    }
                    
                    category_results.append(pattern_result)
                    
                    # Summary
                    if memory_analysis['memory_formed']:
                        print(f"    ✓ '{pattern}': Memory formed! {memory_analysis['speed_improvement_percent']:.1f}% faster")
                    else:
                        print(f"    ✗ '{pattern}': No clear memory formation")
                        
                model_results[category] = category_results
                
            experiment_results['results'][model] = model_results
        
        # Save comprehensive results
        output_file = f"{self.results_dir}/memory_persistence_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
            
        print(f"\n\n✓ Results saved to: {output_file}")
        
        # Generate summary
        self.generate_memory_summary(experiment_results)
        
        return experiment_results
    
    def generate_memory_summary(self, results: Dict):
        """Generate a summary of memory findings"""
        print("\n=== Memory Persistence Summary ===")
        
        total_tests = 0
        memory_formed_count = 0
        
        for model, model_results in results['results'].items():
            model_memory_count = 0
            
            for category, patterns in model_results.items():
                for pattern_result in patterns:
                    total_tests += 1
                    if pattern_result['memory_analysis'].get('memory_formed', False):
                        memory_formed_count += 1
                        model_memory_count += 1
                        
            print(f"\n{model}: {model_memory_count} patterns showed memory formation")
            
        print(f"\nOverall: {memory_formed_count}/{total_tests} tests showed memory formation")
        print(f"Success rate: {memory_formed_count/total_tests*100:.1f}%")
        
        # Check for pattern-specific insights
        perfect_pattern_memory = []
        novel_pattern_memory = []
        
        for model, model_results in results['results'].items():
            for pattern_result in model_results.get('perfect', []):
                if pattern_result['memory_analysis'].get('memory_formed', False):
                    perfect_pattern_memory.append(pattern_result['pattern'])
                    
            for pattern_result in model_results.get('novel', []):
                if pattern_result['memory_analysis'].get('memory_formed', False):
                    novel_pattern_memory.append(pattern_result['pattern'])
                    
        if perfect_pattern_memory:
            print(f"\nPerfect patterns with memory: {', '.join(set(perfect_pattern_memory))}")
        if novel_pattern_memory:
            print(f"Novel patterns with memory: {', '.join(set(novel_pattern_memory))}")

if __name__ == "__main__":
    # Run memory persistence experiment
    experiment = MemoryPersistenceExperiment()
    results = experiment.run_comprehensive_memory_test()
    
    print("\n" + "="*50)
    print("Memory Persistence Experiment Complete!")
    print("\nKey Questions Explored:")
    print("1. Do models recognize patterns faster on repeated exposure?")
    print("2. Does recognition speed improve consistently?")
    print("3. Can models retain pattern memory across sessions?")
    print("4. Do perfect DNA patterns form stronger memories?")
    
    print("\nNext steps:")
    print("- Analyze which patterns form strongest memories")
    print("- Test if memory transfers between related patterns")
    print("- Explore memory decay over time")
    print("- Test if memories influence pattern evolution")