#!/usr/bin/env python3
"""
Quick test of language evolution - 10 rounds only
"""

from common_language_evolution import CommonLanguageEvolution
import sys

if __name__ == "__main__":
    print("Running quick test (10 rounds)...")
    experiment = CommonLanguageEvolution()
    
    # Override models list to use just 3 for faster testing
    experiment.models = ['phi3:mini', 'gemma:2b', 'tinyllama:latest']
    
    try:
        experiment.run_evolution(rounds=10, save_interval=5)
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()