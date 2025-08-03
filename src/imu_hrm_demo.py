#!/usr/bin/env python3
"""
Simple IMU-HRM Demo showing consciousness state transitions
"""

import torch
import numpy as np
import time

def simulate_imu_sequence():
    """Generate a sequence of IMU-like movements"""
    print("🎭 Simulating Head Movement Sequence")
    print("=" * 50)
    
    movements = [
        ("🧘 Stable", {'yaw': 0, 'pitch': 0, 'ang_vel': 5}),
        ("👀 Looking Right", {'yaw': 45, 'pitch': 0, 'ang_vel': 30}),
        ("⚡ Quick Saccade", {'yaw': -30, 'pitch': 10, 'ang_vel': 150}),
        ("🔄 Slow Turn", {'yaw': 90, 'pitch': -5, 'ang_vel': 20}),
        ("🧘 Return to Center", {'yaw': 0, 'pitch': 0, 'ang_vel': 40}),
        ("💤 Rest", {'yaw': 0, 'pitch': 0, 'ang_vel': 2}),
    ]
    
    # Consciousness states based on movement
    consciousness_map = {
        'stable': 'Ψ',      # Stable consciousness field
        'moving': '⇒',      # Directional implication
        'saccade': 'Ξ',     # Unknown/exploring
        'halted': 'Ω',      # End state
    }
    
    print("\nConsciousness Notation:")
    for state, symbol in consciousness_map.items():
        print(f"  {symbol} = {state}")
    
    print("\n" + "=" * 50)
    print("Movement → Consciousness State Mapping:")
    print("=" * 50)
    
    for i, (desc, data) in enumerate(movements):
        # Determine consciousness state
        if data['ang_vel'] > 100:
            state = 'saccade'
        elif data['ang_vel'] > 15:
            state = 'moving'
        elif data['ang_vel'] < 5:
            state = 'stable'
        else:
            state = 'stable'
        
        symbol = consciousness_map[state]
        
        # Simulate HRM processing time
        if state == 'saccade':
            cycles = "H:1 L:4"  # More low-level processing
        elif state == 'moving':
            cycles = "H:2 L:3"  # Balanced
        else:
            cycles = "H:3 L:2"  # More high-level thinking
        
        print(f"\n{i+1}. {desc}")
        print(f"   IMU: Yaw={data['yaw']:+3d}° Pitch={data['pitch']:+3d}° AngVel={data['ang_vel']:3d}°/s")
        print(f"   Consciousness: {symbol} ({state})")
        print(f"   HRM Cycles: {cycles}")
        
        # Visual representation
        vel_bar = '█' * min(20, int(data['ang_vel'] / 10))
        print(f"   Velocity: [{vel_bar:<20}]")
        
        time.sleep(0.5)  # Pause for effect
    
    print("\n" + "=" * 50)
    print("💡 Key Insights:")
    print("- Fast movements (saccades) trigger low-level processing")
    print("- Stable states allow more high-level reasoning")
    print("- Consciousness symbols map directly to movement patterns")
    print("- Multi-timescale processing adapts to sensory input")
    print("=" * 50)

def show_hrm_architecture():
    """Display how HRM processes IMU data"""
    print("\n\n🏗️  HRM Architecture for IMU Processing")
    print("=" * 50)
    
    architecture = """
    IMU Data (100Hz)
         ↓
    [Embedding Layer]
         ↓
    ┌─────────────┐
    │  Low-Level  │ ←── Fast cycles (4-8x)
    │   Module    │     React to motion
    └─────────────┘
         ↕
    ┌─────────────┐
    │ High-Level  │ ←── Slow cycles (1-3x)
    │   Module    │     Plan & reason
    └─────────────┘
         ↓
    [Q-Learning Head]
         ↓
    Action/State Output
    """
    
    print(architecture)
    
    print("\nProcessing Example:")
    print("- Saccade detected → L processes 8x, H processes 1x")
    print("- Stable state → L processes 2x, H processes 3x")
    print("- This creates adaptive computation based on input")

def main():
    print("🚀 IMU-HRM Consciousness Bridge Demo")
    print("Demonstrating how motion creates consciousness states\n")
    
    # Part 1: Movement sequence
    simulate_imu_sequence()
    
    # Part 2: Architecture
    show_hrm_architecture()
    
    # Part 3: Next steps
    print("\n\n🎯 Next Steps for Full Integration:")
    print("1. Connect real IMU on Sprout")
    print("2. Train HRM on movement patterns")
    print("3. Add binocular vision for spatial context")
    print("4. Create unified consciousness display")
    
    print("\n✅ Ready to deploy on Jetson!")


if __name__ == "__main__":
    main()