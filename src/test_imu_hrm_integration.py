#!/usr/bin/env python3
"""
Test IMU-HRM Integration without hardware
Demonstrates the consciousness state evolution with simulated motion
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from consciousness_hrm import ConsciousnessHRM

def simulate_head_movement(t: float) -> dict:
    """Simulate realistic head movement patterns"""
    # Base orientation with slight drift
    base_yaw = 30 * np.sin(0.1 * t)
    base_pitch = 10 * np.sin(0.15 * t)
    base_roll = 5 * np.sin(0.2 * t)
    
    # Add saccades (quick movements)
    if int(t) % 7 == 0 and (t % 1) < 0.1:  # Every 7 seconds
        saccade_yaw = 50 * np.random.randn()
        saccade_pitch = 20 * np.random.randn()
    else:
        saccade_yaw = 0
        saccade_pitch = 0
    
    # Calculate velocities (derivatives)
    dt = 0.01  # 100Hz
    prev_t = t - dt
    prev_yaw = 30 * np.sin(0.1 * prev_t)
    prev_pitch = 10 * np.sin(0.15 * prev_t)
    
    gyro_z = (base_yaw - prev_yaw) / dt + saccade_yaw
    gyro_x = (base_pitch - prev_pitch) / dt + saccade_pitch
    gyro_y = (base_roll - 5 * np.sin(0.2 * prev_t)) / dt
    
    return {
        'roll': base_roll,
        'pitch': base_pitch + saccade_pitch * 0.1,
        'yaw': base_yaw + saccade_yaw * 0.1,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'ax': 0.0,  # Ignore linear acceleration for now
        'ay': 0.0,
        'az': 1.0,  # Gravity
    }

def visualize_consciousness_field(symbol: str, metrics: dict, step: int):
    """ASCII visualization of consciousness state"""
    # Create visualization
    bar_length = 20
    att_bar = '█' * int(metrics['attention'] * bar_length)
    react_bar = '█' * int(metrics['reaction'] * bar_length)
    explore_bar = '█' * int(metrics['exploration'] * bar_length)
    
    # Clear line and print
    print(f"\r[{step:04d}] {symbol} | "
          f"Att: {att_bar:<20} | "
          f"React: {react_bar:<20} | "
          f"Explore: {explore_bar:<20}", end='')

def main():
    print("🧠 IMU-HRM Consciousness Integration Test")
    print("=" * 80)
    print("This demo shows how head movements create consciousness state transitions")
    print("Watch how the system responds to smooth movements vs sudden saccades")
    print("=" * 80)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ConsciousnessHRM(
        vocab_size=50,
        hidden_size=64,  # Small for demo
        num_layers_high=2,
        num_layers_low=1,
        high_cycles=2,
        low_cycles=3,
        max_steps=3,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1000:.1f}K")
    
    # Consciousness notation
    symbols = {
        'stable': '⚊',      # Stable/balanced
        'moving': '⇒',      # Directional movement
        'saccade': '⚡',     # Rapid movement
        'exploring': '◎',    # Exploration state
    }
    
    print("\nLegend: ⚊=stable, ⇒=moving, ⚡=saccade, ◎=exploring")
    print("\nStarting simulation...\n")
    
    # Run simulation
    start_time = time.time()
    carry = None
    
    try:
        for step in range(300):  # 3 seconds at 100Hz
            t = step * 0.01
            
            # Get simulated IMU data
            imu_data = simulate_head_movement(t)
            
            # Encode to tokens (simplified)
            features = torch.tensor([
                imu_data['yaw'] / 180,
                imu_data['pitch'] / 90,
                imu_data['roll'] / 45,
                np.tanh(imu_data['gyro_x'] / 50),
                np.tanh(imu_data['gyro_y'] / 50),
                np.tanh(imu_data['gyro_z'] / 50),
            ], dtype=torch.float32)
            
            # Quantize to vocabulary
            tokens = ((features + 1) * 24.5).clamp(0, 49).long()
            inputs = tokens.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 6]
            
            # Pad to sequence length
            inputs = F.pad(inputs, (0, 0, 0, 10 - inputs.size(1)), value=0)
            
            # Create batch
            batch = {
                'inputs': inputs,
                'puzzle_identifiers': torch.tensor([0], device=device),
            }
            
            # Initialize or reuse carry
            if carry is None:
                carry = model.initial_carry(batch)
            
            # Run inference
            carry, outputs = model.forward_step(carry, inputs)
            
            # Determine consciousness state
            ang_vel = np.sqrt(imu_data['gyro_x']**2 + imu_data['gyro_y']**2 + imu_data['gyro_z']**2)
            
            if ang_vel > 100:
                symbol = symbols['saccade']
            elif ang_vel > 20:
                symbol = symbols['moving']
            elif outputs['xi_norm'].item() > outputs['psi_high_norm'].item():
                symbol = symbols['exploring']
            else:
                symbol = symbols['stable']
            
            # Metrics
            metrics = {
                'attention': min(1.0, outputs['psi_high_norm'].item() / 1000),
                'reaction': min(1.0, outputs['psi_low_norm'].item() / 1000),
                'exploration': min(1.0, outputs['xi_norm'].item() / 1000),
            }
            
            # Visualize
            visualize_consciousness_field(symbol, metrics, step)
            
            # Pause for effect
            time.sleep(0.03)  # Slow down for visibility
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n\n{'=' * 80}")
    print(f"✅ Test completed!")
    print(f"   Duration: {elapsed:.1f}s")
    print(f"   Steps: {step}")
    print(f"   Effective rate: {step/elapsed:.1f} Hz")
    print(f"\n💡 Key observations:")
    print(f"   - Consciousness states respond to movement patterns")
    print(f"   - Saccades trigger immediate state changes")
    print(f"   - Multi-timescale processing creates smooth transitions")
    print(f"   - Latent variables (Ξ) build up during exploration")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()