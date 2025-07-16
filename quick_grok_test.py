#!/usr/bin/env python3
"""
Quick test of Grok's key suggestions
"""

import requests
import json
import time

def test_pattern(pattern: str) -> dict:
    """Test a single pattern across models"""
    models = ["phi3:mini", "tinyllama:latest"]
    results = []
    
    for model in models:
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": model, "prompt": pattern},
                timeout=20
            )
            if response.status_code == 200:
                results.append({
                    'model': model,
                    'success': True,
                    'embedding_size': len(response.json()['embedding'])
                })
        except Exception as e:
            results.append({
                'model': model,
                'success': False,
                'error': str(e)
            })
    
    return {
        'pattern': pattern,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

# Test a few key patterns from Grok's suggestions
test_patterns = [
    # Cross-cultural
    ('chinese_dao', '道'),
    ('sanskrit_om', 'ॐ'),
    ('yin_yang', '☯'),
    # Adversarial
    ('emerge_correct', 'emerge'),
    ('emerge_typo', 'emrge'),
    ('true_caps', 'TRUE'),
]

print("=== Quick Grok Suggestion Test ===\n")

for name, pattern in test_patterns:
    print(f"Testing {name}: '{pattern}'")
    result = test_pattern(pattern)
    
    success_count = sum(1 for r in result['results'] if r['success'])
    print(f"  ✓ {success_count}/{len(result['results'])} models responded\n")
    
    # Save individual result
    with open(f'grok_test_{name}.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

print("\nGrok's suggestions highlight the need for:")
print("1. Cross-cultural pattern testing (道, ॐ, ☯)")
print("2. Adversarial robustness (emrge vs emerge)")
print("3. Neural activation analysis (beyond text)")
print("4. Architecture diversity (beyond transformers)")
print("\nThese insights will shape our next experiments!")