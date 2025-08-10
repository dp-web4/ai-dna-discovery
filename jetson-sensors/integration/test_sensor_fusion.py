#!/usr/bin/env python3
"""
Unified Sensor Fusion Test Script
Integrates IMU, Vision, and Audio sensors with Web4-aligned confidence framework
"""

import sys
import time
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import json
import threading

# Add sensors directory to path
sys.path.append('/home/sprout/ai-workspace/private-context/sensors')

from imu_sensor import IMUSensor
from vision_sensor import VisionSensor
from audio_sensor import AudioSensor

class SensorFusionSystem:
    """Unified sensor fusion implementing reality field concepts"""
    
    def __init__(self):
        # Initialize sensors
        self.imu = IMUSensor()
        self.vision = VisionSensor()
        self.audio = AudioSensor()
        
        # Sensor states
        self.sensors_connected = {
            'imu': False,
            'vision': False,
            'audio': False
        }
        
        # Context state and weights
        self.context = 'stable'
        self.context_history = []
        
        # Dynamic sensor weights based on context
        self.sensor_weights = {
            'stable': {'imu': 0.3, 'vision': 0.5, 'audio': 0.2},
            'moving': {'imu': 0.5, 'vision': 0.4, 'audio': 0.1},
            'turning': {'imu': 0.4, 'vision': 0.5, 'audio': 0.1},
            'unstable': {'imu': 0.6, 'vision': 0.3, 'audio': 0.1}
        }
        
        # Reality field state
        self.reality_field = {}
        self.attention_triggers = []
        
    def connect_sensors(self):
        """Connect all available sensors"""
        print("=" * 60)
        print("SENSOR FUSION SYSTEM INITIALIZATION")
        print("=" * 60)
        
        # Connect IMU
        print("\n[IMU Sensor]")
        if self.imu.connect():
            self.sensors_connected['imu'] = True
            # Try to load calibration
            if not self.imu.load_calibration():
                print("No calibration found - sensor will use defaults")
        else:
            print("IMU connection failed - continuing without IMU")
        
        # Connect Vision
        print("\n[Vision Sensor]")
        if self.vision.connect():
            self.sensors_connected['vision'] = True
        else:
            print("Vision connection failed - continuing without vision")
        
        # Connect Audio
        print("\n[Audio Sensor]")
        if self.audio.connect():
            self.sensors_connected['audio'] = True
        else:
            print("Audio connection failed - continuing without audio")
        
        print("\n" + "=" * 60)
        print("Connected sensors:", [k for k, v in self.sensors_connected.items() if v])
        print("=" * 60 + "\n")
    
    def update_context(self, sensor_data: Dict):
        """Update context based on sensor inputs"""
        # Analyze sensor patterns to determine context
        
        # Check IMU for motion
        if 'imu' in sensor_data and sensor_data['imu']:
            gyro_mag = np.linalg.norm(sensor_data['imu']['data']['angular_velocity'])
            if gyro_mag > 0.5:
                self.context = 'turning'
                return
        
        # Check vision for stability
        if 'vision' in sensor_data and sensor_data['vision']:
            vision_context = sensor_data['vision']['data']['attention']['context']
            if vision_context == 'unstable':
                self.context = 'unstable'
                return
            elif vision_context == 'moving':
                self.context = 'moving'
                return
        
        # Default to stable if no strong signals
        self.context = 'stable'
    
    def check_attention_triggers(self, sensor_data: Dict):
        """Check for attention-triggering events"""
        triggers = []
        
        # Sudden motion trigger (IMU)
        if 'imu' in sensor_data and sensor_data['imu']:
            if sensor_data['imu']['confidence'] < 0.5:
                triggers.append({
                    'type': 'imu_instability',
                    'severity': 'high',
                    'action': 'increase_imu_weight'
                })
        
        # Peripheral instability trigger (Vision)
        if 'vision' in sensor_data and sensor_data['vision']:
            stability = sensor_data['vision']['data']['peripheral_gyroscope']['stability']
            if stability < 0.4:
                triggers.append({
                    'type': 'peripheral_instability',
                    'severity': 'high',
                    'action': 'shift_to_central_vision'
                })
        
        # Audio anomaly trigger
        if 'audio' in sensor_data and sensor_data['audio']:
            if sensor_data['audio']['data']['amplitude'] > 0.8:
                triggers.append({
                    'type': 'loud_sound',
                    'severity': 'medium',
                    'action': 'attend_to_audio'
                })
        
        self.attention_triggers = triggers
        return triggers
    
    def compute_reality_field(self, sensor_data: Dict) -> Dict:
        """Compute unified reality field from all sensors"""
        # Get context-appropriate weights
        weights = self.sensor_weights[self.context]
        
        # Initialize field components
        field = {
            'timestamp': time.time(),
            'context': self.context,
            'sensors': {},
            'overall_confidence': 0,
            'attention_triggers': self.attention_triggers
        }
        
        # Aggregate sensor contributions
        total_weight = 0
        total_confidence = 0
        
        for sensor_type in ['imu', 'vision', 'audio']:
            if sensor_type in sensor_data and sensor_data[sensor_type]:
                data = sensor_data[sensor_type]
                weight = weights[sensor_type]
                
                # Add weighted contribution
                field['sensors'][sensor_type] = {
                    'data': data['data'],
                    'confidence': data['confidence'],
                    'weight': weight,
                    'contribution': data['confidence'] * weight
                }
                
                total_weight += weight
                total_confidence += data['confidence'] * weight
        
        # Calculate overall confidence
        if total_weight > 0:
            field['overall_confidence'] = total_confidence / total_weight
        
        # Store in reality field
        self.reality_field = field
        return field
    
    def display_reality_field(self, field: Dict):
        """Display reality field in human-readable format"""
        print("\n" + "=" * 60)
        print(f"REALITY FIELD | Context: {field['context']} | Confidence: {field['overall_confidence']:.1%}")
        print("=" * 60)
        
        # Display sensor contributions
        for sensor_type, sensor_info in field['sensors'].items():
            print(f"\n[{sensor_type.upper()}] Weight: {sensor_info['weight']:.1f} | Confidence: {sensor_info['confidence']:.1%}")
            
            if sensor_type == 'imu' and 'orientation' in sensor_info['data']:
                euler = sensor_info['data']['orientation']
                print(f"  Orientation (R/P/Y): {euler[0]:.1f}° / {euler[1]:.1f}° / {euler[2]:.1f}°")
                
            elif sensor_type == 'vision' and 'peripheral_gyroscope' in sensor_info['data']:
                stability = sensor_info['data']['peripheral_gyroscope']['stability']
                motion_count = sensor_info['data']['motion']['count']
                print(f"  Peripheral Stability: {stability:.1%} | Motion Regions: {motion_count}")
                
                if 'attention' in sensor_info['data'] and sensor_info['data']['attention']['point']:
                    point = sensor_info['data']['attention']['point']
                    print(f"  Attention Focus: ({point[0]}, {point[1]})")
                
            elif sensor_type == 'audio' and 'amplitude' in sensor_info['data']:
                amp = sensor_info['data']['amplitude']
                freq = sensor_info['data']['frequency']['dominant']
                print(f"  Amplitude: {amp:.4f} | Dominant Freq: {freq:.1f} Hz")
                
                if 'connection' in sensor_info['data'] and sensor_info['data']['connection']['bluetooth_rssi']:
                    rssi = sensor_info['data']['connection']['bluetooth_rssi']
                    latency = sensor_info['data']['connection']['latency_ms']
                    print(f"  Bluetooth RSSI: {rssi} dBm | Latency: {latency:.1f} ms")
        
        # Display attention triggers
        if field['attention_triggers']:
            print(f"\n⚠️  ATTENTION TRIGGERS:")
            for trigger in field['attention_triggers']:
                print(f"  - {trigger['type']} ({trigger['severity']}): {trigger['action']}")
        
        print("=" * 60)
    
    def run_test(self, duration=30):
        """Run sensor fusion test for specified duration"""
        print(f"\nRunning sensor fusion test for {duration} seconds...")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while time.time() - start_time < duration:
                iteration += 1
                
                # Read from all sensors
                sensor_data = {}
                
                if self.sensors_connected['imu']:
                    imu_data = self.imu.get_sensor_fusion_data()
                    if imu_data:
                        sensor_data['imu'] = imu_data
                
                if self.sensors_connected['vision']:
                    vision_data = self.vision.get_sensor_fusion_data()
                    if vision_data:
                        sensor_data['vision'] = vision_data
                
                if self.sensors_connected['audio']:
                    audio_data = self.audio.get_sensor_fusion_data()
                    if audio_data:
                        sensor_data['audio'] = audio_data
                
                # Update context
                self.update_context(sensor_data)
                
                # Check attention triggers
                self.check_attention_triggers(sensor_data)
                
                # Compute reality field
                field = self.compute_reality_field(sensor_data)
                
                # Display every 10th iteration to avoid spam
                if iteration % 10 == 0:
                    self.display_reality_field(field)
                
                # Small delay
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
        
        elapsed = time.time() - start_time
        print(f"\nTest completed - Duration: {elapsed:.1f} seconds")
    
    def save_reality_field(self, filename='reality_field_snapshot.json'):
        """Save current reality field to file"""
        if self.reality_field:
            # Convert numpy arrays to lists for JSON serialization
            field_json = json.dumps(self.reality_field, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
            
            with open(filename, 'w') as f:
                f.write(field_json)
            
            print(f"Reality field saved to {filename}")
    
    def cleanup(self):
        """Clean up sensor connections"""
        print("\nCleaning up sensors...")
        
        if self.sensors_connected['imu']:
            self.imu.close()
        
        if self.sensors_connected['vision']:
            self.vision.close()
        
        if self.sensors_connected['audio']:
            self.audio.close()
        
        print("Sensor fusion system shutdown complete")

def main():
    """Main test function"""
    print("\n" + "█" * 60)
    print(" " * 20 + "SENSOR FUSION TEST")
    print(" " * 15 + "Reality Field Implementation")
    print("█" * 60 + "\n")
    
    # Create sensor fusion system
    fusion = SensorFusionSystem()
    
    # Connect sensors
    fusion.connect_sensors()
    
    # Run test
    fusion.run_test(duration=30)
    
    # Save final reality field
    fusion.save_reality_field()
    
    # Cleanup
    fusion.cleanup()
    
    print("\n" + "█" * 60)
    print(" " * 22 + "TEST COMPLETE")
    print("█" * 60 + "\n")

if __name__ == "__main__":
    main()