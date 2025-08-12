#!/usr/bin/env python3
"""
Run Coherence Engine on Legion (RTX 4090 Linux machine)
Auto-detects available sensors and adapts accordingly
"""

import sys
import os
import time
import threading
from typing import List, Dict, Any

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'plugins/legion'))

# Import sensors
from gpu_sensor import GPUSensor
from audio_sensor import AudioSensor
from camera_sensor import CameraSensor

# Simple coherence engine for now
class CoherenceEngine:
    """Simplified coherence engine for Legion"""
    
    def __init__(self):
        self.sensors = []
        self.context_state = "STABLE"
        self.reality_field = 0.0
        self.tick = 0
        self.running = False
        
        # Trust and relevance weights
        self.trust = {}
        self.relevance = {}
        
    def add_sensor(self, sensor):
        """Add a sensor to the engine"""
        self.sensors.append(sensor)
        self.trust[sensor.id] = 0.5
        self.relevance[sensor.id] = 1.0 / len(self.sensors) if self.sensors else 1.0
        print(f"Added sensor: {sensor.id}")
        
    def update(self):
        """Update reality field from all sensors"""
        self.tick += 1
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        sensor_values = {}
        for sensor in self.sensors:
            try:
                value = sensor.read(tick=self.tick)
                sensor_values[sensor.id] = value
                
                weight = self.trust[sensor.id] * self.relevance[sensor.id]
                weighted_sum += value * weight
                total_weight += weight
                
            except Exception as e:
                print(f"Error reading {sensor.id}: {e}")
                sensor_values[sensor.id] = 0.0
        
        # Calculate reality field
        if total_weight > 0:
            self.reality_field = weighted_sum / total_weight
        else:
            self.reality_field = 0.0
            
        # Update context based on reality field
        if self.reality_field < 0.2:
            self.context_state = "STABLE"
        elif self.reality_field < 0.5:
            self.context_state = "MOVING"
        elif self.reality_field < 0.8:
            self.context_state = "UNSTABLE"
        else:
            self.context_state = "NOVEL"
            
        return sensor_values
        
    def run(self):
        """Main run loop"""
        self.running = True
        
        print("\n" + "="*60)
        print("COHERENCE ENGINE - LEGION")
        print("="*60)
        print(f"Active sensors: {[s.id for s in self.sensors]}")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            while self.running:
                sensor_values = self.update()
                
                # Display status
                print(f"\rTick: {self.tick:5d} | "
                      f"Reality: {self.reality_field:.3f} | "
                      f"Context: {self.context_state:8s} | ", end="")
                
                # Show sensor values
                for sid, val in sensor_values.items():
                    print(f"{sid}: {val:.2f} ", end="")
                
                sys.stdout.flush()
                time.sleep(0.1)  # 10 Hz update rate
                
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.running = False
            
    def cleanup(self):
        """Clean up sensors"""
        for sensor in self.sensors:
            if hasattr(sensor, 'cleanup'):
                sensor.cleanup()


def detect_sensors() -> List:
    """Auto-detect available sensors on Legion"""
    sensors = []
    
    print("Detecting available sensors...")
    
    # Try GPU sensor
    print("  Checking GPU...")
    gpu = GPUSensor()
    if gpu.available:
        sensors.append(gpu)
        stats = gpu.get_detailed_stats()
        if stats.get('available'):
            print(f"    ✓ Found: {stats.get('name', 'NVIDIA GPU')}")
    else:
        print("    ✗ No NVIDIA GPU detected")
    
    # Try camera sensor
    print("  Checking camera...")
    camera = CameraSensor(0)
    if camera.available:
        sensors.append(camera)
        info = camera.get_camera_info()
        print(f"    ✓ Found: Camera {info['index']} ({info['width']}x{info['height']} @ {info['fps']}fps)")
    else:
        print("    ✗ No camera detected")
        
    # Try audio sensor
    print("  Checking audio...")
    audio = AudioSensor()
    if audio.available:
        sensors.append(audio)
        info = audio.get_device_info()
        if info.get('devices'):
            print(f"    ✓ Found: {len(info['devices'])} audio input device(s)")
    else:
        print("    ✗ No audio input detected (using simulated)")
        sensors.append(audio)  # Add anyway for simulated data
    
    return sensors


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("COHERENCE ENGINE FOR LEGION")
    print("Reality Field Generation through Sensor Fusion")
    print("="*60 + "\n")
    
    # Detect available sensors
    sensors = detect_sensors()
    
    if not sensors:
        print("\nNo sensors detected! The engine needs at least one sensor.")
        return 1
        
    print(f"\nDetected {len(sensors)} sensor(s): {[s.id for s in sensors]}")
    
    # Create and configure engine
    engine = CoherenceEngine()
    for sensor in sensors:
        engine.add_sensor(sensor)
    
    # Run the engine
    try:
        engine.run()
    finally:
        engine.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())