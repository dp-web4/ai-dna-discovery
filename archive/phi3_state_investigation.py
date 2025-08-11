#!/usr/bin/env python3
"""
Investigate the state change in Phi3 model
"""

import subprocess
import json
import time
import hashlib
from datetime import datetime

def test_model_response(prompt, model="phi3:mini", attempts=3):
    """Test model response multiple times"""
    responses = []
    
    for i in range(attempts):
        try:
            # Use API for more control
            response = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/generate", "-d",
                 json.dumps({
                     "model": model,
                     "prompt": prompt,
                     "stream": False,
                     "options": {
                         "temperature": 0,
                         "seed": 42,
                         "top_k": 1
                     }
                 })],
                capture_output=True, text=True, timeout=30
            )
            
            if response.returncode == 0:
                result = json.loads(response.stdout)
                text = result.get('response', '').strip()
                responses.append(text)
                print(f"  Attempt {i+1}: '{text}' (len: {len(text)})")
            else:
                print(f"  Attempt {i+1}: ERROR")
                
        except Exception as e:
            print(f"  Attempt {i+1}: Exception - {e}")
            
        time.sleep(2)
    
    return responses

def main():
    print("PHI3 STATE CHANGE INVESTIGATION")
    print("=" * 50)
    
    # Test 1: Check if model is responsive
    print("\nTest 1: Model Responsiveness")
    print("Testing with simple prompts...")
    
    simple_responses = test_model_response("2+2=", attempts=3)
    
    # Test 2: Check the specific prompt that changed
    print("\nTest 2: Reality Definition Test")
    print("Testing 'Define reality in one word.' prompt...")
    
    reality_responses = test_model_response("Define reality in one word.", attempts=5)
    
    # Test 3: Check if it's context-dependent
    print("\nTest 3: Context Reset Test")
    print("Testing after explicit context...")
    
    # First set context
    context_prompt = "You are a philosopher. Define reality in one word."
    context_responses = test_model_response(context_prompt, attempts=3)
    
    # Test 4: Check model info
    print("\nTest 4: Model Status Check")
    try:
        response = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/ps"],
            capture_output=True, text=True
        )
        if response.returncode == 0:
            models = json.loads(response.stdout)
            print(f"  Active models: {len(models.get('models', []))}")
            for model in models.get('models', []):
                print(f"    - {model.get('name')} (expires: {model.get('expires_at')})")
    except:
        print("  Could not check model status")
    
    # Analysis
    print("\n" + "=" * 50)
    print("ANALYSIS:")
    
    # Check consistency
    all_same = all(r == simple_responses[0] for r in simple_responses if r)
    print(f"\nSimple prompt consistency: {'YES' if all_same else 'NO'}")
    
    reality_same = all(r == reality_responses[0] for r in reality_responses if r)
    print(f"Reality prompt consistency: {'YES' if reality_same else 'NO'}")
    
    if not reality_same:
        print("\nResponse variations:")
        for i, r in enumerate(reality_responses):
            print(f"  {i+1}: '{r}'")
    
    # Check for empty responses
    empty_count = sum(1 for r in reality_responses if not r)
    if empty_count > 0:
        print(f"\n⚠️  WARNING: {empty_count}/{len(reality_responses)} empty responses")
        print("This suggests potential model timeout or context issues")
    
    print("\nPOSSIBLE EXPLANATIONS:")
    print("1. Model context window filled during pattern interpretation")
    print("2. Ollama session timeout between requests") 
    print("3. Temperature/seed not being properly applied")
    print("4. Model actually maintaining some state (unlikely based on previous tests)")

if __name__ == "__main__":
    main()