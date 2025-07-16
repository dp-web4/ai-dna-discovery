#!/usr/bin/env python3
"""
Ollama Weight Stability Test
Tests if Ollama model weights remain constant by analyzing embedding consistency
"""

import requests
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import hashlib

class OllamaWeightStabilityTest:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"
        self.test_models = ["phi3:mini", "tinyllama:latest"]
        self.results_dir = '/home/dp/ai-workspace/weight_stability'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def get_embedding_fingerprint(self, model: str, text: str) -> Tuple[np.ndarray, str]:
        """
        Get embedding and create a fingerprint for exact comparison
        """
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={"model": model, "prompt": text},
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = np.array(response.json()['embedding'])
                
                # Create fingerprint using hash of embedding values
                fingerprint = hashlib.sha256(embedding.tobytes()).hexdigest()
                
                return embedding, fingerprint
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            
        return None, None
    
    def test_embedding_stability(self, model: str, pattern: str, num_calls: int = 10):
        """
        Test if same pattern produces identical embeddings across multiple calls
        """
        print(f"\nTesting embedding stability for '{pattern}' on {model}")
        
        embeddings = []
        fingerprints = []
        timestamps = []
        
        for i in range(num_calls):
            start_time = time.time()
            embedding, fingerprint = self.get_embedding_fingerprint(model, pattern)
            elapsed = time.time() - start_time
            
            if embedding is not None:
                embeddings.append(embedding)
                fingerprints.append(fingerprint)
                timestamps.append(elapsed)
                
                # Quick check - are fingerprints identical?
                if i > 0 and fingerprint != fingerprints[0]:
                    print(f"  ⚠️  Embedding changed at call {i+1}!")
                
            time.sleep(0.5)  # Small delay between calls
        
        # Analyze results
        analysis = self.analyze_stability(embeddings, fingerprints, timestamps)
        analysis['pattern'] = pattern
        analysis['model'] = model
        analysis['num_calls'] = num_calls
        
        return analysis
    
    def analyze_stability(self, embeddings: List[np.ndarray], 
                         fingerprints: List[str], 
                         timestamps: List[float]) -> Dict:
        """
        Analyze embedding stability and detect any drift
        """
        if len(embeddings) < 2:
            return {'stable': False, 'reason': 'Insufficient data'}
        
        # Check if all fingerprints are identical
        unique_fingerprints = len(set(fingerprints))
        perfectly_stable = unique_fingerprints == 1
        
        # Calculate embedding similarities
        similarities = []
        for i in range(1, len(embeddings)):
            similarity = np.dot(embeddings[0], embeddings[i]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[i])
            )
            similarities.append(similarity)
        
        # Calculate drift metrics
        max_drift = 1.0 - min(similarities) if similarities else 0
        avg_similarity = np.mean(similarities) if similarities else 1.0
        
        # Timing analysis
        avg_time = np.mean(timestamps)
        time_variance = np.std(timestamps)
        
        return {
            'perfectly_stable': perfectly_stable,
            'unique_fingerprints': unique_fingerprints,
            'min_similarity': float(min(similarities)) if similarities else 1.0,
            'avg_similarity': float(avg_similarity),
            'max_drift': float(max_drift),
            'avg_response_time': float(avg_time),
            'time_variance': float(time_variance),
            'weight_change_detected': not perfectly_stable
        }
    
    def test_memory_effect_on_weights(self, model: str):
        """
        Test if repeated exposure to patterns affects weight stability
        """
        print(f"\n=== Testing Memory Effect on Weight Stability ({model}) ===")
        
        test_sequences = [
            {
                'name': 'Perfect Pattern Repeated',
                'sequence': ['emerge'] * 20  # Same pattern 20 times
            },
            {
                'name': 'Mixed Patterns',
                'sequence': ['emerge', 'true', 'loop', 'know', '∃'] * 4  # Rotating patterns
            },
            {
                'name': 'Novel Pattern Introduction',
                'sequence': ['emerge'] * 10 + ['quantum'] + ['emerge'] * 9  # New pattern in middle
            }
        ]
        
        results = {}
        
        for test in test_sequences:
            print(f"\nTest: {test['name']}")
            sequence_results = []
            
            baseline_embedding, baseline_fingerprint = self.get_embedding_fingerprint(
                model, test['sequence'][0]
            )
            
            for i, pattern in enumerate(test['sequence']):
                embedding, fingerprint = self.get_embedding_fingerprint(model, pattern)
                
                if embedding is not None and baseline_embedding is not None:
                    # Check if this pattern's embedding has changed
                    if pattern == test['sequence'][0]:
                        changed = fingerprint != baseline_fingerprint
                        if changed:
                            print(f"  ! Weight change detected at position {i}")
                    
                    sequence_results.append({
                        'position': i,
                        'pattern': pattern,
                        'fingerprint': fingerprint,
                        'changed_from_baseline': fingerprint != baseline_fingerprint if pattern == test['sequence'][0] else None
                    })
                
                time.sleep(0.2)
            
            results[test['name']] = {
                'sequence_length': len(test['sequence']),
                'unique_patterns': len(set(test['sequence'])),
                'weight_changes_detected': sum(1 for r in sequence_results 
                                             if r.get('changed_from_baseline') == True),
                'results': sequence_results
            }
        
        return results
    
    def run_comprehensive_stability_test(self):
        """
        Run comprehensive weight stability tests
        """
        print("=== Ollama Model Weight Stability Test ===")
        print("Testing if model weights remain constant during inference\n")
        
        test_patterns = {
            'perfect': ['emerge', 'true', '∃'],
            'novel': ['quantum', 'nexus', 'flux'],
            'nonsense': ['xqzt', 'bflm', 'vprw']
        }
        
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'objective': 'Verify if Ollama model weights remain stable during inference',
            'models_tested': self.test_models,
            'tests': {}
        }
        
        for model in self.test_models:
            print(f"\n{'='*50}")
            print(f"Testing model: {model}")
            
            model_results = {
                'embedding_stability': {},
                'memory_effect': None
            }
            
            # Test 1: Basic embedding stability
            for category, patterns in test_patterns.items():
                print(f"\nCategory: {category}")
                category_results = []
                
                for pattern in patterns[:1]:  # Test first pattern in each category
                    result = self.test_embedding_stability(model, pattern, num_calls=5)
                    category_results.append(result)
                    
                    if result['perfectly_stable']:
                        print(f"  ✓ '{pattern}': Perfectly stable (identical embeddings)")
                    else:
                        print(f"  ⚠️  '{pattern}': Detected drift (uniqueness: {result['unique_fingerprints']})")
                
                model_results['embedding_stability'][category] = category_results
            
            # Test 2: Memory effect on weights
            memory_results = self.test_memory_effect_on_weights(model)
            model_results['memory_effect'] = memory_results
            
            full_results['tests'][model] = model_results
        
        # Save results
        output_file = f"{self.results_dir}/stability_test_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Generate summary
        self.generate_stability_summary(full_results)
        
        return full_results
    
    def generate_stability_summary(self, results: Dict):
        """
        Generate summary of weight stability findings
        """
        print("\n=== Weight Stability Summary ===")
        
        for model, model_results in results['tests'].items():
            print(f"\n{model}:")
            
            # Check embedding stability
            stable_count = 0
            total_count = 0
            
            for category, patterns in model_results['embedding_stability'].items():
                for pattern_result in patterns:
                    total_count += 1
                    if pattern_result['perfectly_stable']:
                        stable_count += 1
            
            stability_rate = (stable_count / total_count * 100) if total_count > 0 else 0
            print(f"  Embedding stability: {stability_rate:.1f}% ({stable_count}/{total_count})")
            
            # Check memory effect
            if model_results['memory_effect']:
                total_changes = sum(test['weight_changes_detected'] 
                                  for test in model_results['memory_effect'].values())
                if total_changes == 0:
                    print(f"  Memory effect on weights: None detected")
                else:
                    print(f"  Memory effect on weights: {total_changes} changes detected")
        
        print("\nConclusion:")
        print("If embeddings are perfectly stable → weights are not changing")
        print("If embeddings drift → possible weight updates or numerical instability")

if __name__ == "__main__":
    import os
    
    tester = OllamaWeightStabilityTest()
    
    # Quick test to see if models are responsive
    print("Checking if models are responsive...")
    test_pattern = "test"
    
    for model in tester.test_models[:1]:  # Test first model
        embedding, fingerprint = tester.get_embedding_fingerprint(model, test_pattern)
        if embedding is not None:
            print(f"✓ {model} is responsive")
            print(f"  Embedding dimension: {len(embedding)}")
            print(f"  Fingerprint: {fingerprint[:16]}...")
            
            # Run stability test
            results = tester.run_comprehensive_stability_test()
            
            break
        else:
            print(f"✗ {model} is not responding")
            print("\nModels appear to be offline.")
            print("When models are available, this test will:")
            print("1. Verify if embeddings remain identical across calls")
            print("2. Test if repeated patterns affect weight stability")
            print("3. Detect any weight changes through embedding drift")