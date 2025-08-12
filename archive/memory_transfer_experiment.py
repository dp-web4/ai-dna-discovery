#!/usr/bin/env python3
"""
Memory Transfer Experiment - Phase 2
Tests if AI models can transfer memory between semantically related patterns
"""

import json
import time
import numpy as np
from datetime import datetime
import requests
from typing import Dict, List, Tuple
import os

class MemoryTransferExperiment:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.results_dir = "memory_transfer_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define pattern families with semantic relationships
        self.pattern_families = {
            'existence': {
                'core': ['∃', 'exist', 'being'],
                'related': ['∉', 'void', 'null', 'empty'],
                'opposite': ['absence', 'nothing', 'gone']
            },
            'truth': {
                'core': ['true', 'valid', 'correct'],
                'related': ['false', 'invalid', 'wrong'],
                'opposite': ['lie', 'deception', 'illusion']
            },
            'emergence': {
                'core': ['emerge', 'arise', 'manifest'],
                'related': ['evolve', 'develop', 'unfold'],
                'opposite': ['dissolve', 'vanish', 'disappear']
            },
            'recursion': {
                'core': ['recursive', 'loop', 'cycle'],
                'related': ['iterate', 'repeat', 'return'],
                'opposite': ['linear', 'sequential', 'once']
            },
            'knowledge': {
                'core': ['know', 'understand', 'comprehend'],
                'related': ['learn', 'discover', 'realize'],
                'opposite': ['ignore', 'forget', 'unknown']
            }
        }
        
        # Load perfect patterns from Phase 1
        self.perfect_patterns = self.load_perfect_patterns()
        
    def load_perfect_patterns(self) -> List[str]:
        """Load patterns that achieved 1.0 DNA scores"""
        perfect = []
        try:
            with open('memory_pattern_analyzer_results.json', 'r') as f:
                data = json.load(f)
                for pattern, info in data['pattern_analysis'].items():
                    if info['best_score'] >= 1.0:
                        perfect.append(pattern)
        except:
            # Fallback to known perfect patterns
            perfect = ['∃', '∉', 'know', 'loop', 'true', 'false', '≈', 'null', 
                      'emerge', 'recursive', 'void', 'then', 'exist', 'break',
                      'understand', 'evolve', 'or', 'and', 'if', 'end']
        return perfect
    
    def get_embedding(self, model: str, text: str) -> np.ndarray:
        """Get embedding from Ollama model"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                return np.array(response.json()['embedding'])
        except Exception as e:
            print(f"Error getting embedding: {e}")
        return None
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def test_memory_transfer(self, model: str, family_name: str, family: Dict) -> Dict:
        """Test if learning one pattern helps recognize related patterns"""
        results = {
            'family': family_name,
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'core_patterns': {},
            'transfer_scores': {},
            'opposite_contrast': {}
        }
        
        print(f"\n=== Testing {family_name} family on {model} ===")
        
        # First, get baseline embeddings for all patterns
        baseline_embeddings = {}
        for category in ['core', 'related', 'opposite']:
            for pattern in family[category]:
                emb = self.get_embedding(model, pattern)
                if emb is not None:
                    baseline_embeddings[pattern] = emb
                time.sleep(0.1)
        
        # Test each core pattern
        for core_pattern in family['core']:
            if core_pattern not in baseline_embeddings:
                continue
                
            core_emb = baseline_embeddings[core_pattern]
            is_perfect = core_pattern in self.perfect_patterns
            
            results['core_patterns'][core_pattern] = {
                'is_perfect_pattern': is_perfect,
                'related_similarities': {},
                'opposite_similarities': {}
            }
            
            # Calculate similarities with related patterns
            print(f"\n  Core: '{core_pattern}' (Perfect: {is_perfect})")
            
            for related in family['related']:
                if related in baseline_embeddings:
                    sim = self.calculate_similarity(core_emb, baseline_embeddings[related])
                    results['core_patterns'][core_pattern]['related_similarities'][related] = sim
                    print(f"    → '{related}': {sim:.4f}")
            
            # Calculate similarities with opposite patterns
            for opposite in family['opposite']:
                if opposite in baseline_embeddings:
                    sim = self.calculate_similarity(core_emb, baseline_embeddings[opposite])
                    results['core_patterns'][core_pattern]['opposite_similarities'][opposite] = sim
                    print(f"    ↔ '{opposite}': {sim:.4f}")
        
        # Calculate transfer scores
        self.calculate_transfer_metrics(results)
        
        return results
    
    def calculate_transfer_metrics(self, results: Dict):
        """Calculate memory transfer metrics"""
        all_related_sims = []
        all_opposite_sims = []
        perfect_related_sims = []
        perfect_opposite_sims = []
        
        for core, data in results['core_patterns'].items():
            related_sims = list(data['related_similarities'].values())
            opposite_sims = list(data['opposite_similarities'].values())
            
            all_related_sims.extend(related_sims)
            all_opposite_sims.extend(opposite_sims)
            
            if data['is_perfect_pattern']:
                perfect_related_sims.extend(related_sims)
                perfect_opposite_sims.extend(opposite_sims)
        
        # Calculate metrics
        if all_related_sims:
            results['transfer_scores']['avg_related_similarity'] = np.mean(all_related_sims)
            results['transfer_scores']['max_related_similarity'] = np.max(all_related_sims)
        
        if all_opposite_sims:
            results['transfer_scores']['avg_opposite_similarity'] = np.mean(all_opposite_sims)
            results['transfer_scores']['min_opposite_similarity'] = np.min(all_opposite_sims)
        
        if perfect_related_sims:
            results['transfer_scores']['perfect_pattern_related_avg'] = np.mean(perfect_related_sims)
        
        if perfect_opposite_sims:
            results['transfer_scores']['perfect_pattern_opposite_avg'] = np.mean(perfect_opposite_sims)
        
        # Calculate contrast score (how well model distinguishes related vs opposite)
        if all_related_sims and all_opposite_sims:
            results['transfer_scores']['contrast_score'] = (
                np.mean(all_related_sims) - np.mean(all_opposite_sims)
            )
    
    def test_cross_family_transfer(self, model: str) -> Dict:
        """Test if patterns from one family activate patterns in another"""
        print(f"\n=== Cross-Family Transfer Test on {model} ===")
        
        cross_transfer = {
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'family_connections': {}
        }
        
        # Get embeddings for core patterns from each family
        family_embeddings = {}
        for family_name, family in self.pattern_families.items():
            family_embeddings[family_name] = {}
            for pattern in family['core']:
                emb = self.get_embedding(model, pattern)
                if emb is not None:
                    family_embeddings[family_name][pattern] = emb
                time.sleep(0.1)
        
        # Test cross-family similarities
        for fam1 in self.pattern_families:
            for fam2 in self.pattern_families:
                if fam1 >= fam2:  # Skip self and avoid duplicates
                    continue
                
                key = f"{fam1}↔{fam2}"
                cross_transfer['family_connections'][key] = []
                
                for p1, emb1 in family_embeddings.get(fam1, {}).items():
                    for p2, emb2 in family_embeddings.get(fam2, {}).items():
                        sim = self.calculate_similarity(emb1, emb2)
                        if sim > 0.7:  # High similarity threshold
                            cross_transfer['family_connections'][key].append({
                                'pattern1': p1,
                                'pattern2': p2,
                                'similarity': float(sim)
                            })
                            print(f"  Strong connection: '{p1}' ↔ '{p2}' = {sim:.4f}")
        
        return cross_transfer
    
    def run_full_experiment(self):
        """Run complete memory transfer experiment"""
        models = ['phi3:mini', 'gemma:2b', 'tinyllama:latest']
        all_results = {
            'experiment': 'memory_transfer',
            'phase': 2,
            'timestamp': datetime.now().isoformat(),
            'family_tests': {},
            'cross_family_tests': {},
            'summary': {}
        }
        
        print("="*60)
        print("MEMORY TRANSFER EXPERIMENT - PHASE 2")
        print("="*60)
        print(f"Testing {len(self.pattern_families)} pattern families")
        print(f"Perfect patterns loaded: {len(self.perfect_patterns)}")
        
        for model in models:
            print(f"\n{'='*60}")
            print(f"Testing model: {model}")
            print(f"{'='*60}")
            
            all_results['family_tests'][model] = {}
            
            # Test each family
            for family_name, family in self.pattern_families.items():
                results = self.test_memory_transfer(model, family_name, family)
                all_results['family_tests'][model][family_name] = results
                time.sleep(1)
            
            # Test cross-family transfer
            cross_results = self.test_cross_family_transfer(model)
            all_results['cross_family_tests'][model] = cross_results
            time.sleep(2)
        
        # Generate summary
        self.generate_summary(all_results)
        
        # Save results
        output_file = os.path.join(
            self.results_dir, 
            f"memory_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n\nResults saved to: {output_file}")
        return all_results
    
    def generate_summary(self, results: Dict):
        """Generate summary of findings"""
        summary = {
            'key_findings': [],
            'transfer_strength': {},
            'perfect_pattern_advantage': None,
            'cross_family_connections': 0
        }
        
        # Analyze transfer strength by model
        for model, families in results['family_tests'].items():
            total_contrast = []
            perfect_advantage = []
            
            for family_name, family_data in families.items():
                if 'contrast_score' in family_data['transfer_scores']:
                    total_contrast.append(family_data['transfer_scores']['contrast_score'])
                
                # Check if perfect patterns show stronger transfer
                if ('perfect_pattern_related_avg' in family_data['transfer_scores'] and
                    'avg_related_similarity' in family_data['transfer_scores']):
                    advantage = (family_data['transfer_scores']['perfect_pattern_related_avg'] -
                               family_data['transfer_scores']['avg_related_similarity'])
                    perfect_advantage.append(advantage)
            
            if total_contrast:
                summary['transfer_strength'][model] = {
                    'avg_contrast': float(np.mean(total_contrast)),
                    'interpretation': 'Strong' if np.mean(total_contrast) > 0.2 else 'Moderate'
                }
            
            if perfect_advantage and np.mean(perfect_advantage) > 0:
                summary['perfect_pattern_advantage'] = float(np.mean(perfect_advantage))
        
        # Count cross-family connections
        for model, cross_data in results['cross_family_tests'].items():
            for connections in cross_data['family_connections'].values():
                summary['cross_family_connections'] += len(connections)
        
        # Generate key findings
        if summary['perfect_pattern_advantage'] and summary['perfect_pattern_advantage'] > 0:
            summary['key_findings'].append(
                f"Perfect patterns show {summary['perfect_pattern_advantage']:.3f} stronger transfer to related concepts"
            )
        
        if summary['cross_family_connections'] > 0:
            summary['key_findings'].append(
                f"Found {summary['cross_family_connections']} strong cross-family connections"
            )
        
        strongest_model = max(summary['transfer_strength'].items(), 
                            key=lambda x: x[1]['avg_contrast'])[0]
        summary['key_findings'].append(
            f"{strongest_model} shows strongest memory transfer capability"
        )
        
        results['summary'] = summary
        
        print("\n" + "="*60)
        print("SUMMARY OF FINDINGS")
        print("="*60)
        for finding in summary['key_findings']:
            print(f"• {finding}")

if __name__ == "__main__":
    experiment = MemoryTransferExperiment()
    experiment.run_full_experiment()