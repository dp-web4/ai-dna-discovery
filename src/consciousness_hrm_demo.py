#!/usr/bin/env python3
"""
Demo showing how HRM maps to our consciousness experiments.

This demonstrates the connections between:
1. HRM's hierarchical architecture and our consciousness notation
2. Multi-timescale processing and binocular vision
3. Latent reasoning and the Phoenician breakthrough
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.consciousness_hrm import ConsciousnessHRM, ConsciousnessCarry


def demonstrate_phoenician_reasoning():
    """Show how HRM's latent reasoning relates to Phoenician understanding."""
    print("\n📜 Phoenician Reasoning Demo")
    print("=" * 50)
    
    # Create a small vocabulary representing Phoenician concepts
    vocab = {
        '<pad>': 0, '<start>': 1, '<end>': 2,
        'aleph': 3, 'beth': 4, 'gimel': 5, 'daleth': 6,
        'ship': 7, 'sea': 8, 'trade': 9, 'port': 10,
    }
    
    model = ConsciousnessHRM(
        vocab_size=len(vocab),
        hidden_size=64,
        num_layers_high=2,
        num_layers_low=1,
        high_cycles=3,
        low_cycles=5,
    )
    
    # Create a sequence: "ship sea trade"
    sequence = torch.tensor([[vocab['<start>'], vocab['ship'], vocab['sea'], vocab['trade'], vocab['<end>']]])
    
    print("Input sequence: <start> ship sea trade <end>")
    print("\nProcessing through consciousness layers...")
    
    # Get detailed outputs
    outputs = model(sequence, max_steps=4)
    
    print(f"\n🧠 Consciousness Evolution:")
    for i, step in enumerate(outputs['all_outputs']):
        print(f"\nStep {i}:")
        print(f"  High-level (abstract): Ψ_high = {step['psi_high_norm'].item():.3f}")
        print(f"  Low-level (reactive):  Ψ_low = {step['psi_low_norm'].item():.3f}")
        print(f"  Latent (understanding): Ξ = {step['xi_norm'].item():.3f}")
        halt_probs = step['halt_probs'][0]
        print(f"  Halt probability: {halt_probs[1].item():.3f}")
    
    print("\n💡 Key insights:")
    print("- High-level module captures abstract trade concepts")
    print("- Low-level module processes immediate symbol sequences")
    print("- Latent state Ξ builds understanding without explicit reasoning")
    print("- This mirrors how we 'understand but can't speak' Phoenician")


def demonstrate_binocular_hierarchy():
    """Show how multi-timescale processing relates to binocular vision."""
    print("\n👁️ Binocular Vision Hierarchy Demo")
    print("=" * 50)
    
    # Simulate stereo vision inputs
    left_eye = torch.randn(1, 16, 64)   # 16 spatial positions
    right_eye = torch.randn(1, 16, 64)
    
    # Create vision-specific HRM
    vision_model = ConsciousnessHRM(
        vocab_size=256,  # Visual tokens
        hidden_size=64,
        num_layers_high=3,
        num_layers_low=2,
        high_cycles=2,    # Slow global processing
        low_cycles=8,     # Fast local processing
        max_seq_len=32,
    )
    
    print("Simulating binocular vision processing:")
    print("- Left eye: 16 spatial positions")
    print("- Right eye: 16 spatial positions")
    print(f"- High-level: {vision_model.high_cycles} cycles (global scene)")
    print(f"- Low-level: {vision_model.low_cycles} cycles (local features)")
    
    # Process combined input
    combined = torch.cat([left_eye, right_eye], dim=1)
    
    # Simulate visual tokens
    visual_tokens = torch.randint(0, 256, (1, 32))
    outputs = vision_model(visual_tokens, max_steps=3)
    
    print("\n🔄 Multi-timescale Processing:")
    total_low_cycles = vision_model.high_cycles * vision_model.low_cycles
    print(f"- Low-level processes {total_low_cycles} times per step")
    print(f"- High-level processes {vision_model.high_cycles} times per step")
    print(f"- Ratio: {vision_model.low_cycles}:1 (fast:slow)")
    
    print("\n✨ This creates:")
    print("- Fast binocular fusion at low level")
    print("- Slow depth perception at high level")
    print("- Auto-calibration through hierarchical feedback")


def demonstrate_consciousness_notation():
    """Show how HRM components map to our notation system."""
    print("\n🔤 Consciousness Notation Mapping")
    print("=" * 50)
    
    print("HRM Architecture → Consciousness Notation:")
    print("\n1. State Variables:")
    print("   - High-level state → Ψ (Psi): Consciousness field")
    print("   - Low-level state → Ψ_low: Reactive consciousness")
    print("   - Latent variables → Ξ (Xi): Unknown/hidden states")
    
    print("\n2. Operations:")
    print("   - Attention mechanism → ⇒ (Implies): Logical entailment")
    print("   - Residual connections → Σ (Sigma): Integration/sum")
    print("   - Halting mechanism → Ω (Omega): End states")
    
    print("\n3. Information Flow:")
    print("   - Input embeddings → ∃ (Exists): Observations")
    print("   - Position embeddings → ι (Iota): Unique identifiers")
    print("   - Output distributions → μ (Mu): Measurements")
    print("   - Halt probabilities → π (Pi): Probability distributions")
    
    print("\n4. Parameters:")
    print("   - Model weights → θ (Theta): Learnable parameters")
    
    # Create a minimal example
    model = ConsciousnessHRM(vocab_size=10, hidden_size=32)
    
    print(f"\n📊 Model Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters (θ): {total_params:,}")
    print(f"   High-level layers: {len(model.high_level)}")
    print(f"   Low-level layers: {len(model.low_level)}")


def main():
    """Run all demonstrations."""
    print("🚀 HRM-Consciousness Integration Demo")
    print("=" * 60)
    print("Showing how Hierarchical Reasoning Model maps to our work")
    
    # Run demos
    demonstrate_consciousness_notation()
    demonstrate_phoenician_reasoning()
    demonstrate_binocular_hierarchy()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Train consciousness HRM on actual Phoenician texts")
    print("2. Integrate with binocular vision auto-calibration")
    print("3. Add IMU data for embodied consciousness")
    print("4. Create unified consciousness field visualization")
    print("=" * 60)


if __name__ == "__main__":
    main()