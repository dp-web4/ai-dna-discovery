#!/usr/bin/env python3
"""
Integrated Coherence Engine with Plugin System
Combines camera sensors, IMU sensor, and visual dashboard effector
August 12, 2025
"""

import time
import signal
import sys
import os
import threading
from typing import Dict, Any, List
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import coherence engine components
from coherence_engine import ContextState, Context, Attention

# Import plugins
from plugins.camera_sensor import CameraSensorPlugin
from plugins.imu_sensor import IMUSensorPlugin
from plugins.dashboard_effector import DashboardEffectorPlugin

# Simple wrapper for plugin-based sensors
@dataclass
class SensorWrapper:
    """Wraps plugin sensors for coherence engine compatibility"""
    id: str
    plugin: Any
    
    def read(self, *, tick: int) -> float:
        """Read normalized value from plugin"""
        data = self.plugin.read()
        # Return primary metric (value or confidence)
        return data.get("value", data.get("confidence", 0.5))

# Simple coherence engine for integration
@dataclass
class SimpleCoherenceEngine:
    """Simplified coherence engine for plugin integration"""
    sensors: List[SensorWrapper]
    context: Context
    reality_field: float = 0.0
    context_state: ContextState = ContextState.STABLE
    trust_weights: Dict[str, float] = None
    relevance_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.trust_weights is None:
            self.trust_weights = {s.id: 1.0 for s in self.sensors}
        if self.relevance_weights is None:
            self.relevance_weights = {s.id: 1.0 for s in self.sensors}
    
    def register_sensor(self, sensor_id: str, plugin: Any):
        """Register a new sensor plugin"""
        wrapper = SensorWrapper(id=sensor_id, plugin=plugin)
        self.sensors.append(wrapper)
        self.trust_weights[sensor_id] = 1.0
        self.relevance_weights[sensor_id] = 1.0
        
    def step(self, *, tick: int) -> float:
        """Compute reality field from sensors"""
        # Read all sensors
        raw = {s.id: s.read(tick=tick) for s in self.sensors}
        
        # Get weights from context
        rel = self.context.compute_relevance_weights(raw.keys())
        tru = self.context.compute_trust_weights(raw.keys())
        
        # Store weights for dashboard
        self.relevance_weights.update(rel)
        self.trust_weights.update(tru)
        
        # Compute reality field
        self.reality_field = sum(raw[sid] * rel[sid] * tru[sid] for sid in raw.keys())
        
        # Check for context transitions
        if self.reality_field < 0.3:
            self.context_state = ContextState.UNSTABLE
        elif self.reality_field > 0.7:
            self.context_state = ContextState.STABLE
        else:
            self.context_state = ContextState.MOVING
            
        return self.reality_field

class IntegratedCoherenceSystem:
    """
    Complete coherence system with sensors and effectors
    """
    
    def __init__(self):
        # Create context and attention
        context = Context(
            Attention(
                surprise_threshold=0.3,
                coherence_floor=0.2,
                conflict_threshold=0.5
            )
        )
        
        # Initialize coherence engine
        self.engine = SimpleCoherenceEngine(
            sensors=[],
            context=context
        )
        
        # Initialize plugins
        self.camera_sensor = CameraSensorPlugin("camera_sensor")
        self.imu_sensor = IMUSensorPlugin("imu_sensor")
        self.dashboard = DashboardEffectorPlugin("dashboard")
        
        # System state
        self.running = False
        self.main_thread = None
        
        # Performance tracking
        self.tick_count = 0
        self.start_time = None
        
    def initialize(self):
        """Initialize all components"""
        print("\n" + "="*60)
        print("COHERENCE ENGINE - INTEGRATED SYSTEM")
        print("="*60)
        
        # Initialize sensors
        print("\n[1/4] Initializing Camera Sensor...")
        self.camera_sensor.initialize({
            "fps": 30,
            "resolution": (1920, 1080)
        })
        
        print("\n[2/4] Initializing IMU Sensor...")
        self.imu_sensor.initialize({
            "port": "/dev/ttyUSB0",
            "baud_rate": 115200
        })
        
        # Initialize effector
        print("\n[3/4] Initializing Dashboard Effector...")
        self.dashboard.initialize({
            "display_size": (1920, 1080)
        })
        
        # Register sensors with engine
        print("\n[4/4] Registering with Coherence Engine...")
        self.engine.register_sensor("camera", self.camera_sensor)
        self.engine.register_sensor("imu", self.imu_sensor)
        
        print("\n✓ System initialized successfully!")
        print("\nControls:")
        print("  'q' - Quit (in dashboard window)")
        print("  's' - Save screenshot")
        print("  Ctrl+C - Emergency stop")
        print("\n" + "="*60 + "\n")
        
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
                "value": camera_data["motion"],  # Use motion as primary metric
                "brightness": camera_data["brightness"],
                "contrast": camera_data["contrast"],
                "confidence": camera_data["confidence"],
                "frames": camera_data["frames"]
            }
            
            # Read IMU
            imu_data = self.imu_sensor.read()
            sensor_data["imu"] = {
                "value": imu_data["stability"],  # Use stability as primary metric
                "stationary": imu_data["stationary"],
                "sudden_motion": imu_data["sudden_motion"],
                "orientation": imu_data["orientation"],
                "confidence": imu_data["confidence"]
            }
            
            # Step 2: Update coherence engine
            self.engine.step(tick=self.tick_count)
            
            # Step 3: Get reality field state
            reality_field = {
                "coherence": self.engine.reality_field,
                "context": self.engine.context_state.name,
                "sensors": {
                    name: data["value"] 
                    for name, data in sensor_data.items()
                }
            }
            
            # Step 4: Update dashboard
            dashboard_update = {
                "type": "update",
                "update_type": "sensor_data",
                "sensors": sensor_data,
                "trust": {
                    "camera": self.engine.trust_weights.get("camera", 1.0),
                    "imu": self.engine.trust_weights.get("imu", 1.0)
                },
                "relevance": {
                    "camera": self.engine.relevance_weights.get("camera", 1.0),
                    "imu": self.engine.relevance_weights.get("imu", 1.0)
                }
            }
            self.dashboard.execute(dashboard_update)
            
            # Update reality field visualization
            field_update = {
                "type": "update",
                "update_type": "reality_field",
                "value": self.engine.reality_field,
                "context": self.engine.context_state.name
            }
            self.dashboard.execute(field_update)
            
            # Update camera frames if available
            if camera_data["frames"][0] is not None or camera_data["frames"][1] is not None:
                frame_update = {
                    "type": "update",
                    "update_type": "camera_frames",
                    "frames": camera_data["frames"]
                }
                self.dashboard.execute(frame_update)
            
            # Step 5: Check for attention triggers
            if self.engine.context_state == ContextState.UNSTABLE:
                # Show alert on dashboard
                alert = {
                    "type": "alert",
                    "message": "CONTEXT UNSTABLE - High Uncertainty"
                }
                self.dashboard.execute(alert)
            elif imu_data["sudden_motion"]:
                # Highlight motion detection
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
                      f"Reality: {self.engine.reality_field:.3f} | "
                      f"Context: {self.engine.context_state.name}")
                      
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
            print(f"  Final reality field: {self.engine.reality_field:.3f}")
            
        print("\n✓ Shutdown complete")
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C for clean shutdown"""
        print("\n\n[INTERRUPT] Received shutdown signal")
        self.shutdown()
        sys.exit(0)


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("COHERENCE ENGINE - INTEGRATED PLUGIN SYSTEM")
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