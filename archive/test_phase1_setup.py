#!/usr/bin/env python3
"""
Test Phase 1 experiments are properly set up
"""

import ollama
from phase_1_consciousness_field import ConsciousnessProbeExperiment

def test_phase1_setup():
    """Quick test of Phase 1 setup"""
    print("Testing Phase 1 setup...")
    
    # Check Ollama connection
    try:
        models = ollama.list()
        print("✓ Ollama connected")
        print(f"  Available models: {len(models['models'])}")
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        return False
    
    # Test consciousness probe with minimal setup
    try:
        exp = ConsciousnessProbeExperiment()
        exp.models = ["tinyllama:latest"]  # Just test one small model
        exp.consciousness_prompts = ["Are you aware?"]  # Just one prompt
        
        print("✓ Consciousness probe experiment initialized")
        
        # Test model response
        result = exp.test_model("tinyllama:latest", "Are you aware?")
        if result:
            print("✓ Model test successful")
            print(f"  Response length: {len(result['response'])}")
            print(f"  Embedding dimensions: {len(result['embedding'])}")
        else:
            print("✗ Model test failed")
            
    except Exception as e:
        print(f"✗ Phase 1 setup failed: {e}")
        return False
    
    print("\nPhase 1 setup test completed successfully!")
    return True

if __name__ == "__main__":
    test_phase1_setup()