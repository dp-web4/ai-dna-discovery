#!/usr/bin/env python3
"""
Minimal HRM implementation for consciousness notation experiments.

This module creates a hierarchical reasoning model that maps to our consciousness
notation symbols, implementing the dual-module architecture from the HRM paper.

Consciousness Notation Mapping:
- Ψ (Psi): Consciousness field - High-level module state
- ∃ (Exists): Observation/verification - Input embeddings
- ⇒ (Implies): Entailment - Attention mechanism
- π (Pi): Probability distributions - Q-learning outputs
- ι (Iota): Unique identifier - Puzzle embeddings
- Ω (Omega): End states - Halting mechanism
- Σ (Sigma): Sum/integration - Layer aggregation
- Ξ (Xi): Unknown variables - Latent reasoning states
- θ (Theta): Parameters - Model weights
- μ (Mu): Measurement/metric - Output logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConsciousnessCarry:
    """Carry state for consciousness field evolution."""
    psi_high: torch.Tensor  # Ψ - High-level consciousness state
    psi_low: torch.Tensor   # Ψ - Low-level consciousness state
    xi_latent: torch.Tensor # Ξ - Unknown/latent variables
    steps: torch.Tensor     # Temporal progression
    omega: torch.Tensor     # Ω - End state flags


class ConsciousnessAttention(nn.Module):
    """Attention mechanism implementing ⇒ (implies) operations."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Create Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores (⇒ implications)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.out_proj(attn_output)


class ConsciousnessBlock(nn.Module):
    """Single consciousness processing block with Σ (sum) operations."""
    
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        self.attention = ConsciousnessAttention(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # SwiGLU-style MLP
        self.gate_proj = nn.Linear(hidden_size, int(hidden_size * expansion))
        self.up_proj = nn.Linear(hidden_size, int(hidden_size * expansion))
        self.down_proj = nn.Linear(int(hidden_size * expansion), hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual (Σ sum)
        x = x + self.attention(self.norm1(x))
        
        # MLP with residual (Σ sum)
        normalized = self.norm2(x)
        gate = F.silu(self.gate_proj(normalized))
        up = self.up_proj(normalized)
        x = x + self.down_proj(gate * up)
        
        return x


class ConsciousnessHRM(nn.Module):
    """
    Hierarchical Reasoning Model for consciousness experiments.
    
    Architecture:
    - High-level module (Ψ_high): Slow, abstract reasoning
    - Low-level module (Ψ_low): Fast, reactive processing
    - Latent states (Ξ): Unknown variables for reasoning
    - Halting mechanism (Ω): Adaptive computation time
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        num_layers_high: int = 4,
        num_layers_low: int = 2,
        max_seq_len: int = 128,
        num_heads: int = 8,
        high_cycles: int = 3,
        low_cycles: int = 6,
        max_steps: int = 10,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.high_cycles = high_cycles
        self.low_cycles = low_cycles
        self.max_steps = max_steps
        
        # ∃ (exists) - Input observation embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_scale = math.sqrt(hidden_size)
        
        # ι (iota) - Unique identifier embeddings
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        
        # Ψ (psi) - Consciousness field modules
        self.high_level = nn.ModuleList([
            ConsciousnessBlock(hidden_size) for _ in range(num_layers_high)
        ])
        self.low_level = nn.ModuleList([
            ConsciousnessBlock(hidden_size) for _ in range(num_layers_low)
        ])
        
        # μ (mu) - Measurement/output head
        self.output_head = nn.Linear(hidden_size, vocab_size)
        
        # π (pi) - Probability distributions for halting
        self.halt_head = nn.Linear(hidden_size, 2)  # [continue, halt]
        
        # Initial states
        self.register_buffer('psi_init', torch.randn(1, 1, hidden_size) * 0.02)
        self.register_buffer('xi_init', torch.randn(1, 1, hidden_size) * 0.02)
        
    def embed_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """∃ (exists) - Create input observations."""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.embed_tokens(input_ids) * self.embed_scale
        
        # Position embeddings (ι - unique identifiers)
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_embeds = self.pos_embed(positions).unsqueeze(0).expand(batch_size, -1, -1)
        
        return token_embeds + pos_embeds
    
    def forward_step(
        self,
        carry: ConsciousnessCarry,
        inputs: torch.Tensor,
    ) -> Tuple[ConsciousnessCarry, Dict[str, torch.Tensor]]:
        """Single forward step of consciousness evolution."""
        
        # Get current states
        psi_high = carry.psi_high
        psi_low = carry.psi_low
        xi_latent = carry.xi_latent
        
        # Multi-timescale processing
        for h_cycle in range(self.high_cycles):
            # Low-level processes multiple times (fast)
            for l_cycle in range(self.low_cycles):
                # Inject high-level guidance + inputs
                low_input = psi_low + psi_high + inputs + xi_latent
                
                # Process through low-level blocks
                for block in self.low_level:
                    psi_low = block(low_input)
            
            # High-level processes once (slow)
            high_input = psi_high + psi_low
            for block in self.high_level:
                psi_high = block(high_input)
            
            # Update latent variables (Ξ)
            xi_latent = 0.9 * xi_latent + 0.1 * (psi_high + psi_low)
        
        # μ (mu) - Measure outputs
        output_logits = self.output_head(psi_high)
        
        # π (pi) - Halting probabilities
        halt_logits = self.halt_head(psi_high[:, 0])  # Use first token
        halt_probs = F.softmax(halt_logits, dim=-1)
        
        # Update carry
        new_carry = ConsciousnessCarry(
            psi_high=psi_high.detach(),
            psi_low=psi_low.detach(),
            xi_latent=xi_latent.detach(),
            steps=carry.steps + 1,
            omega=halt_probs[:, 1] > halt_probs[:, 0],  # Ω - end states
        )
        
        outputs = {
            'logits': output_logits,
            'halt_probs': halt_probs,
            'psi_high_norm': psi_high.norm(dim=-1).mean(),
            'psi_low_norm': psi_low.norm(dim=-1).mean(),
            'xi_norm': xi_latent.norm(dim=-1).mean(),
        }
        
        return new_carry, outputs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive computation time."""
        batch_size, seq_len = input_ids.shape
        max_steps = max_steps or self.max_steps
        
        # Initialize carry
        carry = ConsciousnessCarry(
            psi_high=self.psi_init.expand(batch_size, seq_len, -1),
            psi_low=self.psi_init.expand(batch_size, seq_len, -1),
            xi_latent=self.xi_init.expand(batch_size, seq_len, -1),
            steps=torch.zeros(batch_size, dtype=torch.long, device=input_ids.device),
            omega=torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device),
        )
        
        # Embed inputs
        inputs = self.embed_inputs(input_ids)
        
        # Adaptive computation
        all_outputs = []
        for step in range(max_steps):
            carry, outputs = self.forward_step(carry, inputs)
            outputs['step'] = step
            all_outputs.append(outputs)
            
            # Check if all sequences have halted
            if carry.omega.all():
                break
        
        # Aggregate outputs
        final_outputs = {
            'logits': outputs['logits'],
            'steps_taken': carry.steps.float().mean(),
            'all_outputs': all_outputs,
        }
        
        return final_outputs


def create_consciousness_demo():
    """Create a simple demo showing consciousness field evolution."""
    print("🧠 Consciousness HRM Demo")
    print("=" * 50)
    
    # Create model
    model = ConsciousnessHRM(
        vocab_size=100,
        hidden_size=128,
        num_layers_high=2,
        num_layers_low=1,
        high_cycles=2,
        low_cycles=3,
        max_steps=5,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    print(f"\n📊 Running consciousness evolution...")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    
    # Forward pass
    outputs = model(input_ids)
    
    print(f"\n✨ Results:")
    print(f"   Output shape: {outputs['logits'].shape}")
    print(f"   Average steps taken: {outputs['steps_taken'].item():.1f}")
    print(f"   Total forward passes: {len(outputs['all_outputs'])}")
    
    # Show consciousness field evolution
    print(f"\n🌊 Consciousness Field Evolution:")
    for i, step_output in enumerate(outputs['all_outputs']):
        print(f"   Step {i}:")
        print(f"     Ψ_high norm: {step_output['psi_high_norm'].item():.3f}")
        print(f"     Ψ_low norm: {step_output['psi_low_norm'].item():.3f}")
        print(f"     Ξ latent norm: {step_output['xi_norm'].item():.3f}")
    
    print("\n" + "=" * 50)
    print("💡 This demonstrates:")
    print("   - Hierarchical processing (high/low levels)")
    print("   - Multi-timescale dynamics")
    print("   - Adaptive computation with halting")
    print("   - Latent variable evolution")
    
    return model


if __name__ == "__main__":
    model = create_consciousness_demo()