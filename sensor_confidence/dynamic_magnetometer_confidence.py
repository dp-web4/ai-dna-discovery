#!/usr/bin/env python3
"""
Dynamic Magnetometer Confidence Based on Orientation
Implements continuous confidence scaling as IMU tilts
Trust is never absolute - it's always contextual and temporal
"""

import numpy as np
import time
from typing import Dict, Tuple
from dataclasses import dataclass
import json

@dataclass
class OrientationConfidence:
    """Confidence metrics based on device orientation"""
    tilt_angle: float  # Degrees from horizontal
    tilt_rate: float   # Degrees/second
    stability: float   # How stable over recent history
    magnetometer_confidence: float  # 0-1 based on orientation
    timestamp: float

class DynamicMagnetometerConfidence:
    """Compute magnetometer confidence based on real-time orientation"""
    
    def __init__(self, history_window: int = 50):
        self.history = []  # Recent orientation history
        self.max_history = history_window
        
        # Confidence parameters
        self.optimal_tilt = 0.0  # Horizontal is optimal
        self.confidence_at_45deg = 0.5  # 50% confidence at 45° tilt
        self.confidence_at_90deg = 0.15  # 15% confidence when vertical
        self.instability_penalty = 0.3  # Reduce confidence by 30% when unstable
        
    def compute_tilt_angle(self, accel_x: float, accel_y: float, accel_z: float) -> float:
        """Compute tilt angle from horizontal plane"""
        # Normalize acceleration vector
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        if accel_mag < 0.1:  # No gravity detected
            return 90.0  # Assume worst case
            
        # Unit vector
        ax, ay, az = accel_x/accel_mag, accel_y/accel_mag, accel_z/accel_mag
        
        # Angle from vertical (gravity points down in sensor frame)
        # For standard mount: gravity is along -Z axis
        # Tilt angle is deviation from this
        vertical_angle = np.arccos(np.clip(abs(az), -1, 1)) * 180 / np.pi
        
        # Convert to tilt from horizontal
        tilt_from_horizontal = abs(90 - vertical_angle)
        
        return tilt_from_horizontal
    
    def compute_magnetometer_confidence(self, tilt_angle: float) -> float:
        """
        Compute magnetometer confidence based on tilt angle.
        Uses smooth exponential decay from horizontal to vertical.
        """
        # Exponential decay model
        # confidence = exp(-k * tilt_angle)
        # We want: conf(0°) = 1.0, conf(45°) = 0.5, conf(90°) = 0.15
        
        # Solve for k using 45° constraint: 0.5 = exp(-k * 45)
        k = -np.log(0.5) / 45.0  # ≈ 0.0154
        
        # Base confidence from tilt
        base_confidence = np.exp(-k * tilt_angle)
        
        # Ensure minimum confidence at 90°
        if tilt_angle >= 90:
            base_confidence = self.confidence_at_90deg
        
        return np.clip(base_confidence, self.confidence_at_90deg, 1.0)
    
    def compute_stability(self, window_seconds: float = 2.0) -> Tuple[float, float]:
        """
        Compute orientation stability over recent history.
        Returns (stability_score, tilt_rate)
        """
        if len(self.history) < 2:
            return 1.0, 0.0
            
        # Get recent samples within window
        current_time = time.time()
        recent = [h for h in self.history if current_time - h.timestamp <= window_seconds]
        
        if len(recent) < 2:
            return 1.0, 0.0
            
        # Compute tilt rate (degrees/second)
        tilts = [h.tilt_angle for h in recent]
        times = [h.timestamp for h in recent]
        
        # Simple difference for rate
        dt = times[-1] - times[0]
        if dt > 0:
            tilt_rate = abs(tilts[-1] - tilts[0]) / dt
        else:
            tilt_rate = 0.0
            
        # Stability based on standard deviation
        tilt_std = np.std(tilts)
        
        # Convert to 0-1 stability score
        # Low std = high stability
        stability = 1.0 / (1.0 + tilt_std / 10.0)  # 10° std gives 0.5 stability
        
        return stability, tilt_rate
    
    def update_confidence(self, accel_x: float, accel_y: float, accel_z: float,
                         context: Dict = None) -> OrientationConfidence:
        """
        Update magnetometer confidence based on current orientation.
        Returns complete confidence metrics.
        """
        # Compute current tilt
        tilt_angle = self.compute_tilt_angle(accel_x, accel_y, accel_z)
        
        # Get stability metrics
        stability, tilt_rate = self.compute_stability()
        
        # Base confidence from tilt angle
        base_confidence = self.compute_magnetometer_confidence(tilt_angle)
        
        # Apply stability penalty
        if stability < 0.7:  # Unstable
            confidence_multiplier = 1.0 - self.instability_penalty * (1.0 - stability)
            final_confidence = base_confidence * confidence_multiplier
        else:
            final_confidence = base_confidence
            
        # Context modifiers
        if context:
            # Near metal objects
            if context.get('near_metal', False):
                final_confidence *= 0.5
                
            # High acceleration (might not be just gravity)
            accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            if abs(accel_mag - 9.81) > 2.0:  # More than 2 m/s² from gravity
                final_confidence *= 0.7
                
            # Rapid rotation detected
            if context.get('high_rotation', False) or tilt_rate > 30:  # 30°/s
                final_confidence *= 0.8
        
        # Create result
        result = OrientationConfidence(
            tilt_angle=tilt_angle,
            tilt_rate=tilt_rate,
            stability=stability,
            magnetometer_confidence=final_confidence,
            timestamp=time.time()
        )
        
        # Update history
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        return result
    
    def get_confidence_report(self) -> Dict:
        """Generate detailed confidence report"""
        if not self.history:
            return {"error": "No data collected yet"}
            
        recent = self.history[-10:]  # Last 10 samples
        
        report = {
            "current": {
                "tilt_angle": recent[-1].tilt_angle,
                "magnetometer_confidence": recent[-1].magnetometer_confidence,
                "stability": recent[-1].stability,
                "tilt_rate": recent[-1].tilt_rate
            },
            "recent_average": {
                "tilt_angle": np.mean([h.tilt_angle for h in recent]),
                "magnetometer_confidence": np.mean([h.magnetometer_confidence for h in recent]),
                "stability": np.mean([h.stability for h in recent])
            },
            "recommendations": []
        }
        
        # Add recommendations based on current state
        current = recent[-1]
        
        if current.magnetometer_confidence < 0.3:
            report["recommendations"].append(
                "⚠️ Low magnetometer confidence - avoid using for heading"
            )
            
        if current.tilt_angle > 60:
            report["recommendations"].append(
                f"📐 Device tilted {current.tilt_angle:.0f}° from horizontal"
            )
            
        if current.stability < 0.5:
            report["recommendations"].append(
                "🌊 Device unstable - wait for steady state"
            )
            
        if current.tilt_rate > 20:
            report["recommendations"].append(
                f"💨 Rapid movement detected ({current.tilt_rate:.0f}°/s)"
            )
            
        return report

# Integration with existing confidence framework
def create_dynamic_confidence_callback(mag_confidence: DynamicMagnetometerConfidence):
    """Create a callback for the main confidence framework"""
    def compute_magnetometer_confidence(sensor_data: Dict, context: Dict) -> float:
        """Callback to compute magnetometer confidence"""
        result = mag_confidence.update_confidence(
            sensor_data.get('accel_x', 0),
            sensor_data.get('accel_y', 0),
            sensor_data.get('accel_z', 9.81),
            context
        )
        return result.magnetometer_confidence
    
    return compute_magnetometer_confidence

# Demo and testing
def demo_dynamic_confidence():
    """Demonstrate dynamic magnetometer confidence"""
    print("🧭 Dynamic Magnetometer Confidence Demo")
    print("=" * 50)
    
    mag_conf = DynamicMagnetometerConfidence()
    
    # Simulate different orientations
    test_cases = [
        # (name, accel_x, accel_y, accel_z, context)
        ("Horizontal (optimal)", 0, 0, 9.81, {}),
        ("Tilted 30°", 0, 9.81 * np.sin(np.pi/6), 9.81 * np.cos(np.pi/6), {}),
        ("Tilted 45°", 0, 9.81 * np.sin(np.pi/4), 9.81 * np.cos(np.pi/4), {}),
        ("Tilted 60°", 0, 9.81 * np.sin(np.pi/3), 9.81 * np.cos(np.pi/3), {}),
        ("Vertical", 0, 9.81, 0, {}),
        ("Tilted 45° near metal", 0, 9.81 * np.sin(np.pi/4), 9.81 * np.cos(np.pi/4), 
         {"near_metal": True}),
        ("Horizontal but accelerating", 2.0, 0, 11.81, {}),
    ]
    
    print("\nTesting various orientations:\n")
    
    for name, ax, ay, az, context in test_cases:
        result = mag_conf.update_confidence(ax, ay, az, context)
        
        print(f"{name}:")
        print(f"  Tilt angle: {result.tilt_angle:.1f}°")
        print(f"  Magnetometer confidence: {result.magnetometer_confidence:.0%}")
        
        if context:
            print(f"  Context: {context}")
            
        # Add visual confidence indicator
        conf_bar = "█" * int(result.magnetometer_confidence * 20)
        print(f"  Confidence: [{conf_bar:<20}]")
        print()
        
        time.sleep(0.1)  # Small delay for history
    
    # Show report
    print("\n📊 Confidence Report:")
    report = mag_conf.get_confidence_report()
    print(json.dumps(report, indent=2))
    
    # Save confidence curve for visualization
    print("\n💾 Saving confidence curve...")
    
    angles = np.linspace(0, 90, 91)
    confidences = [mag_conf.compute_magnetometer_confidence(angle) for angle in angles]
    
    curve_data = {
        "tilt_angles": angles.tolist(),
        "confidence_values": confidences,
        "model": "exponential_decay",
        "parameters": {
            "optimal_tilt": 0,
            "confidence_at_45deg": 0.5,
            "confidence_at_90deg": 0.15
        }
    }
    
    with open("magnetometer_confidence_curve.json", "w") as f:
        json.dump(curve_data, f, indent=2)
        
    print("Confidence curve saved to magnetometer_confidence_curve.json")
    
    # ASCII visualization of confidence curve
    print("\n📈 Magnetometer Confidence vs Tilt Angle:")
    print("    1.0 |")
    
    for conf_level in [1.0, 0.8, 0.6, 0.4, 0.2]:
        line = f"    {conf_level:.1f} |"
        for angle in range(0, 91, 3):
            conf = mag_conf.compute_magnetometer_confidence(angle)
            if abs(conf - conf_level) < 0.1:
                line += "*"
            else:
                line += " "
        print(line)
        
    print("    0.0 +------------------------------")
    print("        0°    30°    60°    90°")
    print("           Tilt from horizontal")

if __name__ == "__main__":
    demo_dynamic_confidence()