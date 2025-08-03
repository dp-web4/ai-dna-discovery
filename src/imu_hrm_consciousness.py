#!/usr/bin/env python3
"""
IMU-HRM Consciousness Bridge
Integrates IMU orientation data with HRM for embodied consciousness experiments
"""

import sys
import os
import time
import numpy as np
import torch
from collections import deque
from typing import Dict, Tuple, Optional

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imu'))

# Import our modules
from src.consciousness_hrm import ConsciousnessHRM, ConsciousnessCarry
from imu.yahboom_cmp10a import CMP10ADecoder

class IMUHRMBridge:
    """Bridges IMU sensor data to HRM consciousness model"""
    
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=115200):
        # Initialize IMU
        self.imu = CMP10ADecoder(serial_port, baud_rate)
        
        # Initialize HRM
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Using device: {self.device}")
        
        # Smaller model for edge deployment
        self.model = ConsciousnessHRM(
            vocab_size=50,      # Reduced for IMU states
            hidden_size=128,    # Smaller for Jetson
            num_layers_high=2,
            num_layers_low=1,
            high_cycles=2,      # Faster inference
            low_cycles=4,
            max_steps=5,
        ).to(self.device)
        
        # State tracking
        self.orientation_history = deque(maxlen=100)  # 1 second at 100Hz
        self.consciousness_state = None
        
        # Consciousness notation mapping
        self.state_symbols = {
            'stable': 'Ψ',      # Stable consciousness
            'turning': '⇒',     # Implies movement
            'saccade': 'Ξ',     # Unknown/exploring
            'halted': 'Ω',      # End state
        }
        
    def encode_imu_state(self, imu_data: Dict) -> torch.Tensor:
        """Convert IMU data to HRM input format"""
        # Extract key features
        roll = imu_data.get('roll', 0) / 180.0  # Normalize to [-1, 1]
        pitch = imu_data.get('pitch', 0) / 180.0
        yaw = imu_data.get('yaw', 0) / 180.0
        
        # Angular velocities (if available)
        gyro_x = imu_data.get('gyro_x', 0) / 100.0  # Normalize
        gyro_y = imu_data.get('gyro_y', 0) / 100.0
        gyro_z = imu_data.get('gyro_z', 0) / 100.0
        
        # Detect motion states
        is_stable = abs(gyro_x) < 0.1 and abs(gyro_y) < 0.1 and abs(gyro_z) < 0.1
        is_saccade = abs(gyro_x) > 0.5 or abs(gyro_y) > 0.5 or abs(gyro_z) > 0.5
        
        # Create feature vector (10 dimensions)
        features = torch.tensor([
            roll, pitch, yaw,                    # Orientation
            gyro_x, gyro_y, gyro_z,             # Angular velocity
            float(is_stable),                    # Binary states
            float(is_saccade),
            np.sin(yaw * np.pi),                # Cyclic encoding
            np.cos(yaw * np.pi),
        ], dtype=torch.float32)
        
        # Quantize to vocabulary tokens (0-49)
        tokens = ((features + 1) * 24.5).clamp(0, 49).long()
        
        return tokens.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def decode_consciousness_state(self, outputs: Dict) -> str:
        """Decode HRM output to consciousness notation"""
        # Get halt probability
        halt_probs = outputs.get('halt_probs', torch.tensor([0.5, 0.5]))
        is_halted = halt_probs[0, 1] > halt_probs[0, 0]
        
        # Analyze consciousness field norms
        psi_high = outputs.get('psi_high_norm', 0).item()
        psi_low = outputs.get('psi_low_norm', 0).item()
        xi_norm = outputs.get('xi_norm', 0).item()
        
        # Determine state
        if is_halted:
            return self.state_symbols['halted']
        elif xi_norm > psi_high:
            return self.state_symbols['saccade']
        elif psi_low > psi_high * 1.5:
            return self.state_symbols['turning']
        else:
            return self.state_symbols['stable']
    
    def process_step(self, imu_data: Dict) -> Tuple[str, Dict]:
        """Process one IMU reading through HRM"""
        # Encode IMU state
        inputs = self.encode_imu_state(imu_data)
        
        # Create batch format
        batch = {
            'inputs': inputs,
            'puzzle_identifiers': torch.tensor([0], device=self.device),
        }
        
        # Run HRM inference
        if self.consciousness_state is None:
            carry = self.model.initial_carry(batch)
        else:
            carry = self.consciousness_state
        
        # Single step (could do multiple for deeper reasoning)
        carry, outputs = self.model.forward_step(carry, inputs)
        self.consciousness_state = carry
        
        # Decode to consciousness notation
        symbol = self.decode_consciousness_state(outputs)
        
        # Extract control signals
        control = {
            'symbol': symbol,
            'should_halt': carry.omega[0].item(),
            'attention_level': outputs['psi_high_norm'].item(),
            'reaction_speed': outputs['psi_low_norm'].item(),
            'exploration': outputs['xi_norm'].item(),
        }
        
        return symbol, control
    
    def run(self):
        """Main loop integrating IMU with HRM"""
        print("🚀 Starting IMU-HRM Consciousness Bridge")
        print("=" * 50)
        print("Symbols: Ψ=stable, ⇒=turning, Ξ=exploring, Ω=halted")
        print("Press Ctrl+C to exit")
        print("=" * 50)
        
        try:
            # Start IMU reading
            self.imu.start()
            time.sleep(0.1)  # Let IMU stabilize
            
            step_count = 0
            start_time = time.time()
            
            while True:
                # Get latest IMU data
                imu_data = self.imu.get_latest_data()
                
                if imu_data:
                    # Process through HRM
                    symbol, control = self.process_step(imu_data)
                    
                    # Display state
                    if step_count % 10 == 0:  # Every 10 steps
                        elapsed = time.time() - start_time
                        fps = step_count / elapsed if elapsed > 0 else 0
                        
                        print(f"\rStep {step_count} | "
                              f"State: {symbol} | "
                              f"Att: {control['attention_level']:.2f} | "
                              f"React: {control['reaction_speed']:.2f} | "
                              f"Explore: {control['exploration']:.2f} | "
                              f"FPS: {fps:.1f}", end='')
                    
                    step_count += 1
                    
                    # Add to history
                    self.orientation_history.append({
                        'time': time.time(),
                        'imu': imu_data,
                        'consciousness': symbol,
                        'control': control,
                    })
                
                # Control loop rate (~100Hz)
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n✋ Stopping...")
        finally:
            self.imu.stop()
            self.imu.close()
            
            # Summary
            print("\n" + "=" * 50)
            print("📊 Session Summary:")
            print(f"   Total steps: {step_count}")
            print(f"   Duration: {time.time() - start_time:.1f}s")
            print(f"   Average FPS: {step_count / (time.time() - start_time):.1f}")
            
            # State distribution
            if self.orientation_history:
                symbols = [h['consciousness'] for h in self.orientation_history]
                for sym, name in self.state_symbols.items():
                    count = symbols.count(name)
                    pct = 100 * count / len(symbols) if symbols else 0
                    print(f"   {name} ({sym}): {pct:.1f}%")


def main():
    """Demo entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='IMU-HRM Consciousness Bridge')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--test', action='store_true', help='Test without IMU')
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode with simulated data
        print("🧪 Running in test mode with simulated IMU data")
        bridge = IMUHRMBridge(args.port, args.baud)
        
        # Simulate some IMU readings
        test_data = [
            {'roll': 0, 'pitch': 0, 'yaw': 0, 'gyro_x': 0, 'gyro_y': 0, 'gyro_z': 0},
            {'roll': 10, 'pitch': 5, 'yaw': 45, 'gyro_x': 20, 'gyro_y': 10, 'gyro_z': 50},
            {'roll': -5, 'pitch': -10, 'yaw': 90, 'gyro_x': 100, 'gyro_y': 80, 'gyro_z': 120},
            {'roll': 0, 'pitch': 0, 'yaw': 180, 'gyro_x': 5, 'gyro_y': 5, 'gyro_z': 5},
        ]
        
        for i, data in enumerate(test_data):
            symbol, control = bridge.process_step(data)
            print(f"Test {i}: {data} → {symbol} (control: {control})")
    else:
        # Real IMU mode
        bridge = IMUHRMBridge(args.port, args.baud)
        bridge.run()


if __name__ == "__main__":
    main()