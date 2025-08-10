#!/usr/bin/env python3
"""
Coherence Engine - Reality Field Generation through Sensor Fusion
"""

import time
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np
from datetime import datetime

from sensors.base_sensor import SensorReading

@dataclass
class Context:
    """Current context state"""
    state: str  # stable, moving, unstable, novel
    attention_level: float  # 0.0 to 1.0
    confidence: float  # Overall confidence
    active_sensors: List[str]
    timestamp: float

@dataclass
class RealityField:
    """The emergent reality from sensor fusion"""
    timestamp: float
    context: Context
    sensor_contributions: Dict[str, float]  # sensor_name -> weighted contribution
    overall_confidence: float
    attention_triggers: List[Dict[str, Any]]
    predictions: Dict[str, Any]
    raw_readings: Dict[str, SensorReading]

class CoherenceEngine:
    """Main engine that creates reality from sensor fusion"""
    
    def __init__(self, memory_path: str = "memory"):
        self.memory_path = memory_path
        self.sensors = {}  # name -> sensor instance
        self.current_context = Context(
            state="stable",
            attention_level=0.3,
            confidence=0.5,
            active_sensors=[],
            timestamp=time.time()
        )
        
        # Context-based sensor weights
        self.context_weights = {
            "stable": {
                "vision": 0.6,
                "imu": 0.2,
                "memory": 0.1,
                "cognition": 0.1
            },
            "moving": {
                "vision": 0.4,
                "imu": 0.4,
                "memory": 0.1,
                "cognition": 0.1
            },
            "unstable": {
                "vision": 0.2,
                "imu": 0.3,
                "memory": 0.2,
                "cognition": 0.3
            },
            "novel": {
                "vision": 0.3,
                "imu": 0.1,
                "memory": 0.3,
                "cognition": 0.3
            }
        }
        
        # History tracking
        self.reality_history = deque(maxlen=100)
        self.attention_history = deque(maxlen=50)
        
        # Attention trigger thresholds
        self.attention_thresholds = {
            "sudden_change": 0.3,  # Change in any sensor > threshold
            "low_confidence": 0.4,  # Overall confidence < threshold
            "expectation_violation": 0.5,  # Prediction error > threshold
            "sensor_conflict": 0.4  # Disagreement between sensors
        }
        
        # Initialize memory structure
        self._init_memory_structure()
        
    def _init_memory_structure(self):
        """Create memory directory structure"""
        dirs = [
            os.path.join(self.memory_path, "experiences"),
            os.path.join(self.memory_path, "patterns"),
            os.path.join(self.memory_path, "context"),
            os.path.join(self.memory_path, "patterns", "spatial"),
            os.path.join(self.memory_path, "patterns", "temporal"),
            os.path.join(self.memory_path, "patterns", "contextual"),
            os.path.join(self.memory_path, "patterns", "emergent"),
            os.path.join(self.memory_path, "context", "stable"),
            os.path.join(self.memory_path, "context", "unstable"),
            os.path.join(self.memory_path, "context", "transitions")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def register_sensor(self, sensor):
        """Register a new sensor with the engine"""
        self.sensors[sensor.name] = sensor
        self.current_context.active_sensors.append(sensor.name)
        print(f"Registered sensor: {sensor.name} ({sensor.sensor_type})")
        
        # Initialize sensor
        if sensor.initialize():
            sensor.is_active = True
            print(f"  ✓ {sensor.name} initialized")
        else:
            print(f"  ✗ {sensor.name} initialization failed")
    
    def read_all_sensors(self) -> Dict[str, SensorReading]:
        """Read from all active sensors"""
        readings = {}
        
        for name, sensor in self.sensors.items():
            if sensor.is_active:
                try:
                    reading = sensor.read()
                    if reading:
                        readings[name] = reading
                        sensor.update_trust(True, 0.01)
                    else:
                        sensor.update_trust(False, 0.02)
                except Exception as e:
                    print(f"Error reading {name}: {e}")
                    sensor.update_trust(False, 0.05)
        
        return readings
    
    def detect_attention_triggers(self, readings: Dict[str, SensorReading]) -> List[Dict[str, Any]]:
        """Check for conditions that should trigger attention shift"""
        triggers = []
        
        # Check for sudden changes
        for name, reading in readings.items():
            if self.sensors[name].last_reading:
                # Compare with last reading (simplified)
                if reading.confidence < self.sensors[name].last_reading.confidence * 0.7:
                    triggers.append({
                        "type": "sudden_change",
                        "sensor": name,
                        "severity": "high",
                        "details": f"Confidence dropped from {self.sensors[name].last_reading.confidence:.2f} to {reading.confidence:.2f}"
                    })
        
        # Check overall confidence
        overall_conf = np.mean([r.confidence for r in readings.values()]) if readings else 0
        if overall_conf < self.attention_thresholds["low_confidence"]:
            triggers.append({
                "type": "low_confidence",
                "severity": "medium",
                "details": f"Overall confidence {overall_conf:.2f} below threshold"
            })
        
        # Check for sensor conflicts (simplified - comparing confidence levels)
        if len(readings) > 1:
            confidences = [r.confidence for r in readings.values()]
            if max(confidences) - min(confidences) > self.attention_thresholds["sensor_conflict"]:
                triggers.append({
                    "type": "sensor_conflict",
                    "severity": "medium",
                    "details": f"Sensor confidence spread: {min(confidences):.2f} to {max(confidences):.2f}"
                })
        
        return triggers
    
    def update_context(self, readings: Dict[str, SensorReading], triggers: List[Dict[str, Any]]):
        """Update context based on sensor readings and triggers"""
        old_state = self.current_context.state
        
        # Simple context state machine
        if triggers:
            # High attention triggers
            high_severity = any(t["severity"] == "high" for t in triggers)
            if high_severity:
                self.current_context.state = "unstable"
                self.current_context.attention_level = min(1.0, self.current_context.attention_level + 0.3)
            else:
                self.current_context.attention_level = min(1.0, self.current_context.attention_level + 0.1)
        else:
            # Decay attention if no triggers
            self.current_context.attention_level = max(0.1, self.current_context.attention_level - 0.05)
            
            # Update state based on sensor patterns
            if "imu" in readings and readings["imu"].data.get("motion_detected", False):
                self.current_context.state = "moving"
            elif self.current_context.attention_level < 0.3:
                self.current_context.state = "stable"
        
        # Check for novel situations (no matching memories)
        if "memory" in readings:
            memory_confidence = readings["memory"].confidence
            if memory_confidence < 0.3:  # Low memory match = novel
                self.current_context.state = "novel"
        
        # Update context timestamp
        self.current_context.timestamp = time.time()
        
        # Log context transitions
        if old_state != self.current_context.state:
            self._save_context_transition(old_state, self.current_context.state, triggers)
    
    def compute_reality_field(self, readings: Dict[str, SensorReading]) -> RealityField:
        """Compute the reality field from sensor readings"""
        # Get context-appropriate weights
        weights = self.context_weights.get(self.current_context.state, self.context_weights["stable"])
        
        # Calculate weighted contributions
        contributions = {}
        total_weight = 0
        weighted_confidence = 0
        
        for name, reading in readings.items():
            sensor_type = self.sensors[name].sensor_type
            weight = weights.get(sensor_type, 0.1)
            
            # Adjust weight by sensor trust
            trust = self.sensors[name].trust_score
            adjusted_weight = weight * trust
            
            # Calculate contribution
            contribution = reading.confidence * adjusted_weight
            contributions[name] = contribution
            
            total_weight += adjusted_weight
            weighted_confidence += contribution
        
        # Normalize confidence
        overall_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
        
        # Update context confidence
        self.current_context.confidence = overall_confidence
        
        # Generate predictions (simplified for now)
        predictions = self._generate_predictions(readings)
        
        # Detect attention triggers
        triggers = self.detect_attention_triggers(readings)
        
        # Create reality field
        field = RealityField(
            timestamp=time.time(),
            context=self.current_context,
            sensor_contributions=contributions,
            overall_confidence=overall_confidence,
            attention_triggers=triggers,
            predictions=predictions,
            raw_readings=readings
        )
        
        return field
    
    def _generate_predictions(self, readings: Dict[str, SensorReading]) -> Dict[str, Any]:
        """Generate predictions based on current state"""
        predictions = {}
        
        # Ask temporal sensors for predictions
        for name, sensor in self.sensors.items():
            if hasattr(sensor, 'predict_future'):
                try:
                    prediction = sensor.predict_future(timesteps=5)
                    if prediction:
                        predictions[name] = prediction
                except:
                    pass
        
        return predictions
    
    def _save_context_transition(self, old_state: str, new_state: str, triggers: List):
        """Save context transition to memory"""
        transition = {
            "timestamp": datetime.now().isoformat(),
            "old_state": old_state,
            "new_state": new_state,
            "triggers": triggers,
            "active_sensors": self.current_context.active_sensors
        }
        
        # Save to context/transitions
        filename = f"{self.memory_path}/context/transitions/{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(transition, f, indent=2)
    
    def save_experience(self, reality_field: RealityField):
        """Save current experience to memory"""
        # Create daily directory
        date_str = datetime.now().strftime("%Y-%m-%d")
        day_path = os.path.join(self.memory_path, "experiences", date_str)
        os.makedirs(day_path, exist_ok=True)
        
        # Save timestamped experience
        timestamp_str = datetime.now().strftime("%H-%M-%S")
        filename = os.path.join(day_path, f"{timestamp_str}.json")
        
        # Convert to serializable format
        experience = {
            "timestamp": reality_field.timestamp,
            "context": asdict(reality_field.context),
            "sensor_contributions": reality_field.sensor_contributions,
            "overall_confidence": reality_field.overall_confidence,
            "attention_triggers": reality_field.attention_triggers,
            "predictions": reality_field.predictions
        }
        
        with open(filename, 'w') as f:
            json.dump(experience, f, indent=2)
    
    def step(self) -> RealityField:
        """Single step of the coherence engine"""
        # Read all sensors
        readings = self.read_all_sensors()
        
        # Compute reality field
        reality_field = self.compute_reality_field(readings)
        
        # Update context based on the field
        self.update_context(readings, reality_field.attention_triggers)
        
        # Store in history
        self.reality_history.append(reality_field)
        
        # Save experience periodically (every 10 steps)
        if len(self.reality_history) % 10 == 0:
            self.save_experience(reality_field)
        
        # Update sensor last readings
        for name, reading in readings.items():
            self.sensors[name].last_reading = reading
        
        return reality_field
    
    def run(self, duration: int = 60):
        """Run the coherence engine for specified duration"""
        print(f"\nCoherence Engine starting...")
        print(f"Active sensors: {list(self.sensors.keys())}")
        print(f"Initial context: {self.current_context.state}")
        print("-" * 60)
        
        start_time = time.time()
        step_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Single step
                reality_field = self.step()
                step_count += 1
                
                # Display status every 10 steps
                if step_count % 10 == 0:
                    self.display_status(reality_field)
                
                # Small delay
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping engine...")
        
        print(f"\nCoherence Engine stopped after {step_count} steps")
        print(f"Final context: {self.current_context.state}")
        print(f"Experiences saved to: {self.memory_path}/experiences/")
    
    def display_status(self, reality_field: RealityField):
        """Display current engine status"""
        print(f"\n[Step {len(self.reality_history)}] Context: {self.current_context.state} | "
              f"Attention: {self.current_context.attention_level:.2f} | "
              f"Confidence: {reality_field.overall_confidence:.2f}")
        
        # Show sensor contributions
        if reality_field.sensor_contributions:
            print("  Sensors:", end=" ")
            for name, contrib in reality_field.sensor_contributions.items():
                print(f"{name}: {contrib:.3f}", end=" | ")
            print()
        
        # Show triggers
        if reality_field.attention_triggers:
            print("  ⚠ Triggers:", end=" ")
            for trigger in reality_field.attention_triggers:
                print(f"{trigger['type']} ({trigger['severity']})", end=" | ")
            print()
    
    def shutdown(self):
        """Clean shutdown of engine and sensors"""
        print("\nShutting down Coherence Engine...")
        
        # Save final experience
        if self.reality_history:
            self.save_experience(self.reality_history[-1])
        
        # Shutdown sensors
        for name, sensor in self.sensors.items():
            sensor.shutdown()
            print(f"  Shutdown {name}")
        
        print("Coherence Engine shutdown complete")