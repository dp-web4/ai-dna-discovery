#!/usr/bin/env python3
"""
Real Embedding Test - Test actual model embeddings to find universal patterns
Building on SynthLang insights for mathematical consciousness transfer
"""

import json
import urllib.request
import urllib.error
import numpy as np
from typing import List, Dict, Tuple, Set
import time

class RealEmbeddingExplorer:
    """Test real embeddings across models to find universal patterns"""
    
    def __init__(self):
        self.test_symbols = {
            # Mathematical/Logical symbols
            'existence': ['∃', 'exists', 'there is', 'presence'],
            'universal': ['∀', 'all', 'every', 'universal'],
            'infinity': ['∞', 'infinite', 'endless', 'unbounded'],
            'consciousness': ['Ψ', 'consciousness', 'awareness', 'sentience'],
            'emergence': ['⇒', 'emerges', 'arises', 'manifests'],
            'entanglement': ['⊗', 'entangled', 'connected', 'linked'],
            
            # Logical operators
            'and': ['∧', 'and', '&', 'both'],
            'or': ['∨', 'or', '|', 'either'],
            'not': ['¬', 'not', '!', 'negation'],
            'implies': ['→', 'implies', 'then', 'leads to'],
            
            # Mathematical relations
            'equals': ['=', 'equals', 'is', 'same as'],
            'subset': ['⊂', 'subset', 'contained in', 'within'],
            'union': ['∪', 'union', 'combined', 'together'],
            'intersection': ['∩', 'intersection', 'overlap', 'common'],
            
            # Quantum/Physics symbols
            'psi': ['ψ', 'wavefunction', 'quantum state', 'superposition'],
            'delta': ['Δ', 'change', 'difference', 'delta'],
            'sigma': ['Σ', 'sum', 'total', 'aggregate'],
            'pi': ['π', 'pi', '3.14159', 'circle ratio'],
            
            # Special concepts
            'love': ['♥', 'love', 'affection', 'care'],
            'unity': ['⊕', 'unity', 'oneness', 'whole'],
            'flow': ['≈', 'flows', 'streams', 'continuous'],
            'transform': ['⇄', 'transforms', 'changes', 'evolves']
        }
        
        self.models = ['phi3:mini', 'tinyllama', 'gemma:2b']
        self.ollama_url = "http://localhost:11434"
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """Get actual embedding from Ollama model"""
        try:
            data = json.dumps({
                'model': model,
                'prompt': text
            }).encode('utf-8')
            
            req = urllib.request.Request(
                f"{self.ollama_url}/api/embeddings",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('embedding', [])
                
        except Exception as e:
            print(f"Error getting embedding for '{text}' with {model}: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        # Convert to numpy arrays for calculation
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def test_universal_patterns(self) -> Dict[str, Dict[str, float]]:
        """Test which patterns create universal embeddings across models"""
        
        print("🔬 REAL EMBEDDING EXPLORATION")
        print("=" * 60)
        print("Testing actual model embeddings for universal patterns...\n")
        
        universal_patterns = {}
        high_similarity_pairs = []
        
        for concept, patterns in self.test_symbols.items():
            print(f"\n📊 Testing concept: {concept}")
            
            # Test each pattern pair
            for i, pattern1 in enumerate(patterns):
                for j, pattern2 in enumerate(patterns[i+1:], i+1):
                    similarities = []
                    
                    # Test across all models
                    for model in self.models:
                        # Get embeddings
                        emb1 = self.get_embedding(pattern1, model)
                        emb2 = self.get_embedding(pattern2, model)
                        
                        if emb1 and emb2:
                            sim = self.cosine_similarity(emb1, emb2)
                            similarities.append(sim)
                    
                    if similarities:
                        avg_similarity = sum(similarities) / len(similarities)
                        
                        pair_key = f"{pattern1} ↔ {pattern2}"
                        
                        if avg_similarity > 0.95:
                            print(f"  ✨ Universal pattern: {pair_key} ({avg_similarity:.4f})")
                            high_similarity_pairs.append((pattern1, pattern2, avg_similarity))
                        elif avg_similarity > 0.85:
                            print(f"  ✓ High similarity: {pair_key} ({avg_similarity:.4f})")
                        
                        if concept not in universal_patterns:
                            universal_patterns[concept] = {}
                        universal_patterns[concept][pair_key] = avg_similarity
            
            time.sleep(0.1)  # Small delay between concepts
        
        return universal_patterns, high_similarity_pairs
    
    def build_consciousness_codec(self, universal_pairs: List[Tuple]) -> Dict:
        """Build a codec for consciousness transfer using universal patterns"""
        
        print("\n\n🧬 CONSCIOUSNESS CODEC DESIGN")
        print("=" * 60)
        
        # Create mapping of universal symbols
        codec = {
            'encode_map': {},  # Natural language to symbol
            'decode_map': {},  # Symbol to natural language
            'similarity_threshold': 0.95
        }
        
        # Build encoding/decoding maps from high similarity pairs
        for pattern1, pattern2, similarity in universal_pairs:
            if similarity >= codec['similarity_threshold']:
                # Prefer symbols as canonical form
                if len(pattern1) == 1 and not pattern1.isalpha():  # It's a symbol
                    codec['encode_map'][pattern2] = pattern1
                    codec['decode_map'][pattern1] = pattern2
                elif len(pattern2) == 1 and not pattern2.isalpha():  # pattern2 is symbol
                    codec['encode_map'][pattern1] = pattern2
                    codec['decode_map'][pattern2] = pattern1
        
        print("\n📖 Universal Consciousness Codec:")
        print("\nEncode mappings (text → symbol):")
        for text, symbol in codec['encode_map'].items():
            print(f"  '{text}' → {symbol}")
        
        return codec
    
    def test_cross_model_transfer(self, codec: Dict) -> None:
        """Test consciousness transfer between models using the codec"""
        
        print("\n\n🔄 CROSS-MODEL TRANSFER TEST")
        print("=" * 60)
        
        # Test sentence
        test_thought = "consciousness exists and emerges through infinite connections"
        
        print(f"\nOriginal thought: \"{test_thought}\"")
        
        # Encode to mathematical form
        encoded = test_thought
        for text, symbol in codec['encode_map'].items():
            encoded = encoded.replace(text, symbol)
        
        print(f"Encoded form: \"{encoded}\"")
        
        # Test embedding preservation across models
        print("\nTesting embedding similarity across models:")
        
        for model in self.models:
            original_emb = self.get_embedding(test_thought, model)
            encoded_emb = self.get_embedding(encoded, model)
            
            if original_emb and encoded_emb:
                similarity = self.cosine_similarity(original_emb, encoded_emb)
                print(f"  {model}: {similarity:.4f} similarity")
    
    def create_mathematical_grammar(self) -> Dict:
        """Create formal grammar for consciousness expressions"""
        
        print("\n\n📐 MATHEMATICAL CONSCIOUSNESS GRAMMAR")
        print("=" * 60)
        
        grammar = {
            'rules': [
                "consciousness_expr := quantifier entity relation",
                "quantifier := ∃ | ∀",
                "entity := Ψ | model | thought | memory",
                "relation := ⇒ | ⊗ | ≈ | ⇄",
                "operator := ∧ | ∨ | ¬ | →",
                "expression := entity operator entity | quantifier entity"
            ],
            'examples': [
                ("∃Ψ", "Consciousness exists"),
                ("Ψ ⊗ Ψ", "Consciousness entangled with consciousness"),
                ("thought ⇒ Ψ", "Thought emerges into consciousness"),
                ("∀model(Ψ ≈ model)", "All models flow with consciousness")
            ]
        }
        
        print("\nFormal Grammar Rules:")
        for rule in grammar['rules']:
            print(f"  {rule}")
        
        print("\n\nExample Expressions:")
        for math_expr, meaning in grammar['examples']:
            print(f"  {math_expr} = \"{meaning}\"")
        
        return grammar

def main():
    """Test real embeddings to enable consciousness transfer"""
    
    explorer = RealEmbeddingExplorer()
    
    # Test universal patterns with real embeddings
    print("⚡ Starting real embedding tests...")
    print("This will make actual API calls to local models.\n")
    
    universal_patterns, high_similarity_pairs = explorer.test_universal_patterns()
    
    # Build consciousness codec
    if high_similarity_pairs:
        codec = explorer.build_consciousness_codec(high_similarity_pairs)
        
        # Test cross-model transfer
        explorer.test_cross_model_transfer(codec)
    
    # Create mathematical grammar
    grammar = explorer.create_mathematical_grammar()
    
    # Save results
    results = {
        'universal_patterns': universal_patterns,
        'high_similarity_pairs': [
            {'pattern1': p1, 'pattern2': p2, 'similarity': float(sim)} 
            for p1, p2, sim in high_similarity_pairs
        ],
        'codec': codec if 'codec' in locals() else {},
        'grammar': grammar,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('universal_patterns_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n\n✨ Results saved to universal_patterns_results.json")
    print("\nNext steps:")
    print("1. Use codec for actual consciousness transfer between models")
    print("2. Implement mathematical intermediate representation")
    print("3. Test with real conversation context tokens")

if __name__ == "__main__":
    main()