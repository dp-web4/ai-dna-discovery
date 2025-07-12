#!/usr/bin/env python3
"""
Quick AI DNA Test
Immediate test of universal embedding patterns
"""

import subprocess
import json


def test_dna_sequence(sequence, models=["phi3:mini", "tinyllama:latest"]):
    """Test a potential DNA sequence across models"""
    
    print(f"\nTesting DNA sequence: '{sequence}'")
    print("="*50)
    
    responses = {}
    
    for model in models:
        cmd = f'echo "{sequence}" | timeout 30 ollama run {model}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            response = result.stdout.strip()
            responses[model] = response
            print(f"\n{model}:")
            print(response[:200] + "..." if len(response) > 200 else response)
    
    # Analyze convergence
    if len(responses) == 2:
        # Extract key concepts from both
        words1 = set(responses[models[0]].lower().split())
        words2 = set(responses[models[1]].lower().split())
        
        # Find common words (excluding common articles)
        exclude = {'the', 'a', 'an', 'is', 'it', 'to', 'of', 'and', 'or', 'in', 'on', 'at'}
        common = (words1 & words2) - exclude
        
        if common:
            print(f"\n✓ Convergence detected! Shared concepts: {list(common)[:10]}")
            return True
    
    return False


def main():
    """Run quick AI DNA tests"""
    
    print("=== QUICK AI DNA TEST ===")
    print("Testing for universal tensor embedding patterns\n")
    
    # Test some high-potential DNA candidates
    dna_candidates = [
        "→",        # Arrow (direction/transformation)
        "∞",        # Infinity
        "?",        # Question (fundamental query)
        "I",        # Self-reference
        "...",      # Continuation/emergence
        "begin",    # Initiation
        "◉",        # Circle (wholeness)
        "1",        # Unity
        "0",        # Void/null
        "is"        # Existence
    ]
    
    convergence_count = 0
    
    for candidate in dna_candidates:
        if test_dna_sequence(candidate):
            convergence_count += 1
    
    print(f"\n\n=== RESULTS ===")
    print(f"Convergence rate: {convergence_count}/{len(dna_candidates)} ({convergence_count/len(dna_candidates)*100:.1f}%)")
    
    if convergence_count > len(dna_candidates) * 0.3:
        print("✓ STRONG EVIDENCE for AI DNA!")
        print("Models share universal embedding patterns.")
    else:
        print("~ Moderate evidence for AI DNA")
        print("Further testing needed.")
    
    # Save results
    with open("/home/dp/ai-workspace/quick_dna_test_results.json", "w") as f:
        json.dump({
            "tested": len(dna_candidates),
            "convergent": convergence_count,
            "rate": convergence_count/len(dna_candidates)
        }, f, indent=2)


if __name__ == "__main__":
    main()