#!/usr/bin/env python3
"""
Quick test of Grok's suggestions with our current models
"""

import requests
import json
import numpy as np
from typing import Dict, List
import time

class GrokSuggestionTester:
    def __init__(self):
        self.base_url = "http://localhost:11434/api/embeddings"
        self.models = ["phi3:mini", "tinyllama:latest", "gemma:2b"]
        
        # Cross-cultural test patterns
        self.cultural_patterns = {
            'chinese_dao': '道',
            'sanskrit_om': 'ॐ',
            'arabic_one': 'واحد',
            'infinity': '∞',
            'yin_yang': '☯',
        }
        
        # Adversarial patterns
        self.adversarial_tests = {
            'emerge': ['emerge', 'emrge', 'emmerge', 'Emerge', 'EMERGE'],
            'true': ['true', 'tru', 'True', 'TRUE', 'truee'],
            'loop': ['loop', 'lop', 'Loop', 'loops', 'l00p'],
        }
    
    def get_embedding(self, model: str, text: str) -> List[float]:
        """Get embedding from a model"""
        try:
            response = requests.post(
                self.base_url,
                json={"model": model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()['embedding']
        except:
            pass
        return None
    
    def calculate_dna_score(self, embeddings: List[List[float]]) -> float:
        """Calculate similarity across embeddings"""
        if not all(embeddings):
            return 0.0
        
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = self.cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity with dimension handling"""
        # Ensure same dimensions by truncating to min length
        min_len = min(len(a), len(b))
        a_np = np.array(a[:min_len])
        b_np = np.array(b[:min_len])
        
        # Avoid division by zero
        norm_product = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        if norm_product == 0:
            return 0.0
            
        return np.dot(a_np, b_np) / norm_product
    
    def test_cultural_patterns(self):
        """Test cross-cultural patterns"""
        print("=== Testing Cross-Cultural Patterns ===\n")
        results = {}
        
        for name, pattern in self.cultural_patterns.items():
            embeddings = []
            for model in self.models:
                emb = self.get_embedding(model, pattern)
                if emb:
                    embeddings.append(emb)
            
            score = self.calculate_dna_score(embeddings)
            results[name] = {
                'pattern': pattern,
                'score': score,
                'models_tested': len(embeddings)
            }
            
            print(f"{name} '{pattern}': {score:.3f}")
            if score >= 0.8:
                print(f"  ⚡ FUNDAMENTAL PATTERN DISCOVERED!")
        
        return results
    
    def test_adversarial_robustness(self):
        """Test pattern robustness"""
        print("\n=== Testing Adversarial Robustness ===\n")
        results = {}
        
        for base_pattern, variants in self.adversarial_tests.items():
            print(f"\nTesting '{base_pattern}' variants:")
            variant_scores = []
            
            for variant in variants:
                embeddings = []
                for model in self.models:
                    emb = self.get_embedding(model, variant)
                    if emb:
                        embeddings.append(emb)
                
                score = self.calculate_dna_score(embeddings)
                variant_scores.append({
                    'variant': variant,
                    'score': score
                })
                print(f"  '{variant}': {score:.3f}")
            
            results[base_pattern] = variant_scores
        
        return results

if __name__ == "__main__":
    tester = GrokSuggestionTester()
    
    # Test cultural patterns
    cultural_results = tester.test_cultural_patterns()
    
    # Test adversarial robustness
    adversarial_results = tester.test_adversarial_robustness()
    
    # Save results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cultural_patterns': cultural_results,
        'adversarial_tests': adversarial_results
    }
    
    with open('grok_suggestions_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Results saved to grok_suggestions_results.json")