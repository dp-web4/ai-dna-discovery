#!/usr/bin/env python3
"""
Simplified Integrated Coherence System
Direct integration without complex dependencies
August 12, 2025
"""

import time
import signal
import sys
import os
import threading
import numpy as np
from typing import Dict, Any, List
from enum import Enum, auto

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import plugins
from plugins.camera_sensor import CameraSensorPlugin
from plugins.imu_sensor import IMUSensorPlugin
from plugins.dashboard_effector import DashboardEffectorPlugin

# Simple context states
class ContextState(Enum):
    STABLE = auto()
    MOVING = auto()
    UNSTABLE = auto()
    NOVEL = auto()

class IntegratedCoherenceSystem:
    """
    Complete coherence system with sensors and effectors
    """
    
    def __init__(self):
        # System state
        self.reality_field = 0.0
        self.context_state = ContextState.STABLE
        self.trust_weights = {}
        self.relevance_weights = {}
        
        # Initialize plugins
        self.camera_sensor = CameraSensorPlugin("camera_sensor")
        self.imu_sensor = IMUSensorPlugin("imu_sensor")
        self.dashboard = DashboardEffectorPlugin("dashboard")
        
        # System control
        self.running = False
        self.tick_count = 0
        self.start_time = None
        
        # History for stability detection
        self.field_history = []
        
    def initialize(self):
        """Initialize all components"""
        print("\n" + "="*60)
        print("COHERENCE ENGINE - INTEGRATED SYSTEM")
        print("="*60)
        
        # Initialize sensors
        print("\n[1/3] Initializing Camera Sensor...")
        self.camera_sensor.initialize({
            "fps": 30,
            "resolution": (1920, 1080)
        })
        
        print("\n[2/3] Initializing IMU Sensor...")
        self.imu_sensor.initialize({
            "port": "/dev/ttyUSB0",
            "baud_rate": 115200
        })
        
        # Initialize effector
        print("\n[3/3] Initializing Dashboard Effector...")
        self.dashboard.initialize({
            "display_size": (1920, 1080)
        })
        
        # Initialize weights
        self.trust_weights = {
            "camera": 1.0,
            "imu": 1.0
        }
        self.relevance_weights = {
            "camera": 1.0,
            "imu": 1.0
        }
        
        print("\n✓ System initialized successfully!")
        print("\nControls:")
        print("  'q' - Quit (in dashboard window)")
        print("  's' - Save screenshot")
        print("  Ctrl+C - Emergency stop")
        print("\n" + "="*60 + "\n")
        
    def compute_context_weights(self) -> Dict[str, float]:
        """Compute relevance weights based on context"""
        if self.context_state == ContextState.STABLE:
            # In stable context, trust vision more
            return {"camera": 1.0, "imu": 0.5}
        elif self.context_state == ContextState.MOVING:
            # When moving, balance both
            return {"camera": 0.8, "imu": 0.8}
        elif self.context_state == ContextState.UNSTABLE:
            # When unstable, trust IMU more
            return {"camera": 0.5, "imu": 1.0}
        else:  # NOVEL
            # In novel situations, use all sensors equally
            return {"camera": 1.0, "imu": 1.0}
            
    def update_context(self, sensor_data: Dict[str, Any]):
        """Update context based on sensor readings"""
        # Check for sudden motion
        if sensor_data["imu"]["sudden_motion"]:
            self.context_state = ContextState.UNSTABLE
            return
            
        # Check for stability
        if sensor_data["imu"]["stationary"] and sensor_data["camera"]["value"] < 0.1:
            self.context_state = ContextState.STABLE
            return
            
        # Check for movement
        if sensor_data["camera"]["value"] > 0.3 or not sensor_data["imu"]["stationary"]:
            self.context_state = ContextState.MOVING
            return
            
        # Check field history for novelty
        if len(self.field_history) > 10:
            recent_std = np.std(self.field_history[-10:])
            if recent_std > 0.3:
                self.context_state = ContextState.NOVEL
                
    def compute_reality_field(self, sensor_data: Dict[str, Any]) -> float:
        """Compute reality field from sensor data"""
        field = 0.0
        
        # Get sensor values
        camera_val = sensor_data["camera"]["value"]
        camera_conf = sensor_data["camera"]["confidence"]
        imu_val = sensor_data["imu"]["value"]
        imu_conf = sensor_data["imu"]["confidence"]
        
        # Apply weights
        camera_contrib = camera_val * camera_conf * \
                        self.trust_weights["camera"] * \
                        self.relevance_weights["camera"]
        imu_contrib = imu_val * imu_conf * \
                     self.trust_weights["imu"] * \
                     self.relevance_weights["imu"]
                     
        # Normalize
        total_weight = (self.trust_weights["camera"] * self.relevance_weights["camera"] + 
                       self.trust_weights["imu"] * self.relevance_weights["imu"])
        
        if total_weight > 0:
            field = (camera_contrib + imu_contrib) / total_weight
            
        return min(max(field, 0.0), 1.0)  # Clamp to [0, 1]
        
    def update_trust(self, sensor_data: Dict[str, Any]):
        """Update trust weights based on sensor consistency"""
        # Simple trust update: reduce trust if sensors disagree
        camera_motion = sensor_data["camera"]["value"] > 0.3
        imu_motion = not sensor_data["imu"]["stationary"]
        
        if camera_motion != imu_motion:
            # Sensors disagree, reduce trust slightly
            self.trust_weights["camera"] *= 0.99
            self.trust_weights["imu"] *= 0.99
        else:
            # Sensors agree, increase trust slightly
            self.trust_weights["camera"] = min(1.0, self.trust_weights["camera"] * 1.01)
            self.trust_weights["imu"] = min(1.0, self.trust_weights["imu"] * 1.01)
            
    def run(self):
        """Main coherence loop"""
        self.running = True
        self.start_time = time.time()
        
        # Set up signal handler for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print("Starting coherence engine...")
        
        while self.running and self.dashboard.running:
            tick_start = time.time()
            
            # Step 1: Read all sensors
            sensor_data = {}
            
            # Read camera
            camera_data = self.camera_sensor.read()
            sensor_data["camera"] = {
                "value": camera_data.get("motion", 0.0),
                "brightness": camera_data.get("brightness", 0.5),
                "contrast": camera_data.get("contrast", 0.5),
                "confidence": camera_data.get("confidence", 0.0),
                "frames": camera_data.get("frames", [None, None])
            }
            
            # Read IMU
            imu_data = self.imu_sensor.read()
            sensor_data["imu"] = {
                "value": imu_data.get("stability", 0.5),
                "stationary": imu_data.get("stationary", False),
                "sudden_motion": imu_data.get("sudden_motion", False),
                "orientation": imu_data.get("orientation", [0, 0, 0]),
                "confidence": imu_data.get("confidence", 0.5)
            }
            
            # Step 2: Update context and weights
            self.update_context(sensor_data)
            self.relevance_weights = self.compute_context_weights()
            self.update_trust(sensor_data)
            
            # Step 3: Compute reality field
            self.reality_field = self.compute_reality_field(sensor_data)
            self.field_history.append(self.reality_field)
            if len(self.field_history) > 100:
                self.field_history.pop(0)
                
            # Step 4: Update dashboard
            dashboard_update = {
                "type": "update",
                "update_type": "sensor_data",
                "sensors": sensor_data,
                "trust": self.trust_weights.copy(),
                "relevance": self.relevance_weights.copy()
            }
            self.dashboard.execute(dashboard_update)
            
            # Update reality field visualization
            field_update = {
                "type": "update",
                "update_type": "reality_field",
                "value": self.reality_field,
                "context": self.context_state.name
            }
            self.dashboard.execute(field_update)
            
            # Update camera frames - always send them for continuous display
            frame_update = {
                "type": "update",
                "update_type": "camera_frames",
                "frames": sensor_data["camera"]["frames"]
            }
            self.dashboard.execute(frame_update)
            
            # Step 5: Check for alerts
            if self.context_state == ContextState.UNSTABLE:
                alert = {
                    "type": "alert",
                    "message": "CONTEXT UNSTABLE"
                }
                self.dashboard.execute(alert)
            elif sensor_data["imu"]["sudden_motion"]:
                alert = {
                    "type": "alert",
                    "message": "SUDDEN MOTION DETECTED"
                }
                self.dashboard.execute(alert)
                
            # Step 6: Performance tracking
            self.tick_count += 1
            
            if self.tick_count % 30 == 0:  # Every second at 30 FPS
                elapsed = time.time() - self.start_time
                avg_tick_rate = self.tick_count / elapsed
                print(f"Tick {self.tick_count} | "
                      f"Rate: {avg_tick_rate:.1f} Hz | "
                      f"Reality: {self.reality_field:.3f} | "
                      f"Context: {self.context_state.name} | "
                      f"Trust: C:{self.trust_weights['camera']:.2f} I:{self.trust_weights['imu']:.2f}")
                      
            # Maintain target rate (30 Hz)
            tick_elapsed = time.time() - tick_start
            if tick_elapsed < 0.033:  # 30 FPS
                time.sleep(0.033 - tick_elapsed)
                
    def shutdown(self):
        """Clean shutdown of all components"""
        print("\n\nShutting down coherence system...")
        
        self.running = False
        
        # Shutdown in reverse order
        print("  Stopping dashboard...")
        self.dashboard.teardown()
        
        print("  Stopping IMU sensor...")
        self.imu_sensor.teardown()
        
        print("  Stopping camera sensor...")
        self.camera_sensor.teardown()
        
        # Print final statistics
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\nSession Statistics:")
            print(f"  Total ticks: {self.tick_count}")
            print(f"  Runtime: {total_time:.1f} seconds")
            print(f"  Average rate: {self.tick_count/total_time:.1f} Hz")
            print(f"  Final reality field: {self.reality_field:.3f}")
            print(f"  Final context: {self.context_state.name}")
            print(f"  Final trust: Camera={self.trust_weights['camera']:.3f}, IMU={self.trust_weights['imu']:.3f}")
            
        print("\n✓ Shutdown complete")
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C for clean shutdown"""
        print("\n\n[INTERRUPT] Received shutdown signal")
        self.shutdown()
        sys.exit(0)


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("COHERENCE ENGINE - SIMPLIFIED INTEGRATION")
    print("Reality emerges from sensor fusion")
    print("="*60)
    
    # Create system
    system = IntegratedCoherenceSystem()
    
    try:
        # Initialize
        system.initialize()
        
        # Run main loop
        system.run()
        
    except Exception as e:
        print(f"\n[ERROR] System error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ensure clean shutdown
        system.shutdown()


if __name__ == "__main__":
    main()