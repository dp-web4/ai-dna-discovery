#!/usr/bin/env python3
"""
Quick Weight Stability Test for Ollama Models
Tests if embeddings remain identical across multiple calls
"""

import requests
import json
import time
import numpy as np
import hashlib
from datetime import datetime

def get_embedding_fingerprint(model: str, text: str):
    """Get embedding and create a fingerprint for exact comparison"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30
        )
        
        if response.status_code == 200:
            embedding = np.array(response.json()['embedding'])
            fingerprint = hashlib.sha256(embedding.tobytes()).hexdigest()
            return embedding, fingerprint
        
    except Exception as e:
        print(f"Error: {e}")
    
    return None, None

def test_weight_stability(model: str = "phi3:mini"):
    """Test if model weights remain constant"""
    print(f"\n=== Weight Stability Test for {model} ===")
    print("Testing if embeddings remain identical across calls...")
    
    test_patterns = {
        'perfect': ['emerge', 'true', '∃'],
        'novel': ['quantum', 'nexus'],
        'nonsense': ['xqzt', 'bflm']
    }
    
    results = {}
    
    for category, patterns in test_patterns.items():
        print(f"\n{category.upper()} Patterns:")
        
        for pattern in patterns:
            print(f"\n  Testing '{pattern}':")
            
            # Get 5 embeddings for the same pattern
            fingerprints = []
            embeddings = []
            
            for i in range(5):
                embedding, fingerprint = get_embedding_fingerprint(model, pattern)
                if embedding is not None:
                    embeddings.append(embedding)
                    fingerprints.append(fingerprint)
                    print(f"    Call {i+1}: {fingerprint[:16]}...")
                time.sleep(0.2)
            
            # Check if all fingerprints are identical
            if len(set(fingerprints)) == 1:
                print(f"    ✓ PERFECTLY STABLE - All embeddings identical")
                stability = "perfect"
            else:
                print(f"    ⚠️  UNSTABLE - {len(set(fingerprints))} different embeddings")
                stability = "unstable"
            
            # Calculate similarity even for "unstable" cases
            if len(embeddings) >= 2:
                similarities = []
                for i in range(1, len(embeddings)):
                    sim = np.dot(embeddings[0], embeddings[i]) / (
                        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[i])
                    )
                    similarities.append(sim)
                avg_similarity = np.mean(similarities)
                print(f"    Average similarity: {avg_similarity:.10f}")
            
            results[pattern] = {
                'category': category,
                'stability': stability,
                'unique_fingerprints': len(set(fingerprints)),
                'fingerprints': fingerprints
            }
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    stable_count = sum(1 for r in results.values() if r['stability'] == 'perfect')
    total_count = len(results)
    
    print(f"Perfectly stable patterns: {stable_count}/{total_count}")
    
    if stable_count == total_count:
        print("\n✓ CONCLUSION: Model weights appear to be STATIC")
        print("  All patterns produce identical embeddings across calls")
        print("  This suggests weights do not change during inference")
    else:
        print("\n⚠️  CONCLUSION: Detected embedding variations")
        print("  Some patterns produce different embeddings")
        print("  This could indicate:")
        print("  - Numerical instability")
        print("  - Dynamic weight updates")
        print("  - Non-deterministic processing")
    
    return results

# Test memory effect
def test_memory_effect(model: str = "phi3:mini"):
    """Test if repeated exposure affects embeddings"""
    print("\n\n=== Memory Effect Test ===")
    print("Testing if repeated exposure changes embeddings...")
    
    pattern = "emerge"
    print(f"\nPattern: '{pattern}'")
    
    # Get baseline
    baseline_embedding, baseline_fingerprint = get_embedding_fingerprint(model, pattern)
    print(f"Baseline fingerprint: {baseline_fingerprint[:16]}...")
    
    # Expose pattern 20 times
    print("\nExposing pattern 20 times...")
    for i in range(20):
        _ = get_embedding_fingerprint(model, pattern)
        if i % 5 == 0:
            print(f"  Exposure {i+1}/20")
        time.sleep(0.1)
    
    # Check if embedding changed
    final_embedding, final_fingerprint = get_embedding_fingerprint(model, pattern)
    print(f"\nFinal fingerprint: {final_fingerprint[:16]}...")
    
    if baseline_fingerprint == final_fingerprint:
        print("✓ No change detected - weights remain stable despite repeated exposure")
    else:
        print("⚠️  Change detected - embeddings differ after repeated exposure")
        similarity = np.dot(baseline_embedding, final_embedding) / (
            np.linalg.norm(baseline_embedding) * np.linalg.norm(final_embedding)
        )
        print(f"Similarity: {similarity:.10f}")

if __name__ == "__main__":
    print("=== Ollama Model Weight Stability Analysis ===")
    print("Objective: Determine if model weights change during inference")
    print("Method: Compare embedding fingerprints for identical inputs")
    
    # Run stability test
    results = test_weight_stability()
    
    # Run memory effect test
    test_memory_effect()
    
    print("\n\nKey Insight for AI DNA Discovery:")
    print("If weights are static, then memory and pattern recognition")
    print("must emerge from the architecture itself, not from weight updates.")