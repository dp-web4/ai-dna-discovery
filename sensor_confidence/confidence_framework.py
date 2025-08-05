#!/usr/bin/env python3
"""
Sensor Confidence Framework
Implements LCT/T3/V3-inspired confidence metrics for embedded sensors
Each sensor evaluates its own trustworthiness and contextual relevance
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
import time
from abc import ABC, abstractmethod

@dataclass
class ConfidenceMetrics:
    """Confidence metrics for a sensor at a given time"""
    raw_confidence: float  # 0-1: How well is the sensor working?
    contextual_relevance: float  # 0-1: How important is this sensor right now?
    historical_reliability: float  # 0-1: How reliable has this sensor been?
    fractal_confidence: float = 0.0  # Computed from above
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Compute fractal confidence from components"""
        # Fractal combination: each level influences the final score
        # but with diminishing weights as we go deeper
        self.fractal_confidence = (
            0.5 * self.raw_confidence +  # Current state matters most
            0.3 * self.contextual_relevance +  # Context is important
            0.2 * self.historical_reliability  # History provides baseline
        )

class SensorConfidence(ABC):
    """Base class for sensor confidence evaluation"""
    
    def __init__(self, name: str, window_size: int = 100):
        self.name = name
        self.confidence_history = deque(maxlen=window_size)
        self.audit_results = {}
        self.last_audit_time = 0
        self.audit_interval = 60.0  # Seconds between audits
        
    @abstractmethod
    def audit(self) -> Dict[str, float]:
        """Perform sensor audit/calibration, return confidence scores"""
        pass
    
    @abstractmethod
    def evaluate_context(self, context: Dict) -> float:
        """Evaluate sensor relevance given current context"""
        pass
    
    @abstractmethod
    def compute_raw_confidence(self, sensor_data: Dict) -> float:
        """Compute confidence from current sensor readings"""
        pass
    
    def get_historical_reliability(self) -> float:
        """Compute reliability from confidence history"""
        if not self.confidence_history:
            return 0.5  # Neutral starting point
        
        # Weight recent history more heavily
        weights = np.exp(-0.1 * np.arange(len(self.confidence_history)))
        weights = weights / weights.sum()
        
        confidences = [m.fractal_confidence for m in self.confidence_history]
        return np.average(confidences, weights=weights)
    
    def update_confidence(self, sensor_data: Dict, context: Dict) -> ConfidenceMetrics:
        """Update confidence metrics based on current data and context"""
        # Check if we need a new audit
        current_time = time.time()
        if current_time - self.last_audit_time > self.audit_interval:
            self.audit_results = self.audit()
            self.last_audit_time = current_time
        
        # Compute current metrics
        raw_conf = self.compute_raw_confidence(sensor_data)
        context_rel = self.evaluate_context(context)
        hist_rel = self.get_historical_reliability()
        
        metrics = ConfidenceMetrics(
            raw_confidence=raw_conf,
            contextual_relevance=context_rel,
            historical_reliability=hist_rel
        )
        
        self.confidence_history.append(metrics)
        return metrics

class IMUConfidence(SensorConfidence):
    """IMU-specific confidence evaluation"""
    
    def __init__(self):
        super().__init__("IMU")
        self.expected_gravity = 9.81
        self.vertical_mount_penalty = 0.7  # Reduce magnetometer confidence
        
    def audit(self) -> Dict[str, float]:
        """Calibrate IMU and assess component confidence"""
        print(f"Auditing {self.name}...")
        
        # In real implementation, would collect samples
        # For now, return example audit results
        audit = {
            'accelerometer': 0.95,  # Usually reliable
            'gyroscope': 0.90,      # Good when not saturated
            'magnetometer': 0.3,    # Low due to vertical mount
            'temperature': 0.85     # Decent
        }
        
        # Check mounting orientation
        # If vertical mount detected, penalize magnetometer
        if self._detect_vertical_mount():
            audit['magnetometer'] *= self.vertical_mount_penalty
            print("Vertical mount detected - magnetometer confidence reduced")
        
        return audit
    
    def _detect_vertical_mount(self) -> bool:
        """Detect if IMU is mounted vertically"""
        # In real implementation, check gravity vector
        # For now, return True as you mentioned vertical mount
        return True
    
    def compute_raw_confidence(self, sensor_data: Dict) -> float:
        """Compute IMU confidence from sensor readings"""
        if not sensor_data:
            return 0.0
        
        confidences = []
        
        # Check accelerometer (gravity magnitude)
        accel_mag = np.sqrt(
            sensor_data.get('accel_x', 0)**2 +
            sensor_data.get('accel_y', 0)**2 +
            sensor_data.get('accel_z', 0)**2
        )
        # Confidence based on how close to expected gravity
        accel_conf = 1.0 - min(abs(accel_mag - self.expected_gravity) / 5.0, 1.0)
        confidences.append(accel_conf * self.audit_results.get('accelerometer', 1.0))
        
        # Check gyroscope (should be near zero when stationary)
        gyro_mag = np.sqrt(
            sensor_data.get('gyro_x', 0)**2 +
            sensor_data.get('gyro_y', 0)**2 +
            sensor_data.get('gyro_z', 0)**2
        )
        # High values might indicate saturation
        gyro_conf = 1.0 - min(gyro_mag / 500.0, 1.0)  # 500 deg/s threshold
        confidences.append(gyro_conf * self.audit_results.get('gyroscope', 1.0))
        
        # Magnetometer confidence (low for vertical mount)
        mag_conf = self.audit_results.get('magnetometer', 0.3)
        confidences.append(mag_conf)
        
        # Combine with weights
        weights = [0.4, 0.4, 0.2]  # Less weight on magnetometer
        return np.average(confidences, weights=weights)
    
    def evaluate_context(self, context: Dict) -> float:
        """Evaluate IMU relevance in current context"""
        relevance = 0.1  # Base relevance
        
        # Motion context
        if context.get('motion_detected', False):
            relevance += 0.7
        
        # Navigation context  
        if context.get('navigation_active', False):
            relevance += 0.5
            
        # Vision stabilization context
        if context.get('vision_active', False):
            relevance += 0.3
            
        return min(relevance, 1.0)

class CameraConfidence(SensorConfidence):
    """Camera-specific confidence evaluation"""
    
    def __init__(self, camera_id: str):
        super().__init__(f"Camera_{camera_id}")
        self.camera_id = camera_id
        self.min_brightness = 10
        self.max_brightness = 245
        
    def audit(self) -> Dict[str, float]:
        """Calibrate camera and assess confidence"""
        print(f"Auditing {self.name}...")
        
        # Example audit results
        audit = {
            'focus': 0.9,
            'exposure': 0.85,
            'color_balance': 0.8,
            'noise_level': 0.7
        }
        
        return audit
    
    def compute_raw_confidence(self, sensor_data: Dict) -> float:
        """Compute camera confidence from image metrics"""
        if not sensor_data:
            return 0.0
            
        confidences = []
        
        # Brightness check
        brightness = sensor_data.get('mean_brightness', 128)
        if brightness < self.min_brightness or brightness > self.max_brightness:
            brightness_conf = 0.3  # Too dark or saturated
        else:
            brightness_conf = 0.9
        confidences.append(brightness_conf)
        
        # Contrast check (standard deviation)
        contrast = sensor_data.get('std_brightness', 0)
        contrast_conf = min(contrast / 50.0, 1.0)  # Good contrast around 50
        confidences.append(contrast_conf)
        
        # Motion blur check
        blur_score = sensor_data.get('blur_score', 0)
        blur_conf = 1.0 - min(blur_score / 100.0, 1.0)
        confidences.append(blur_conf)
        
        return np.mean(confidences)
    
    def evaluate_context(self, context: Dict) -> float:
        """Evaluate camera relevance in current context"""
        relevance = 0.1  # Base relevance
        
        # Visual task active
        if context.get('vision_active', False):
            relevance += 0.8
            
        # Lighting conditions
        if context.get('good_lighting', False):
            relevance += 0.2
        elif context.get('poor_lighting', False):
            relevance -= 0.3
            
        # Multiple cameras - stereo vision
        if context.get('stereo_vision', False) and self.camera_id in ['0', '1']:
            relevance += 0.3
            
        return max(0, min(relevance, 1.0))

class SensorConfidenceManager:
    """Manages confidence for all sensors in the system"""
    
    def __init__(self):
        self.sensors: Dict[str, SensorConfidence] = {}
        self.system_context = {}
        self.confidence_log = []
        
    def add_sensor(self, sensor: SensorConfidence):
        """Add a sensor to the confidence system"""
        self.sensors[sensor.name] = sensor
        # Perform initial audit
        sensor.audit()
        
    def update_context(self, **kwargs):
        """Update system context"""
        self.system_context.update(kwargs)
        
    def update_all_confidence(self, sensor_data: Dict[str, Dict], context: Dict = None) -> Dict[str, ConfidenceMetrics]:
        """Update confidence for all sensors"""
        results = {}
        
        # Use provided context or system context
        if context:
            self.system_context.update(context)
        
        for name, sensor in self.sensors.items():
            data = sensor_data.get(name, {})
            metrics = sensor.update_confidence(data, self.system_context)
            results[name] = metrics
            
        # Log results
        self.confidence_log.append({
            'timestamp': time.time(),
            'metrics': {name: m.__dict__ for name, m in results.items()},
            'context': self.system_context.copy()
        })
        
        return results
    
    def get_system_confidence(self) -> float:
        """Compute overall system confidence (fractal of fractals)"""
        if not self.sensors:
            return 0.0
            
        # Weight sensors by their contextual relevance
        total_weight = 0
        weighted_confidence = 0
        
        for name, sensor in self.sensors.items():
            if sensor.confidence_history:
                latest = sensor.confidence_history[-1]
                weight = latest.contextual_relevance
                weighted_confidence += weight * latest.fractal_confidence
                total_weight += weight
                
        if total_weight > 0:
            return weighted_confidence / total_weight
        return 0.0
    
    def save_confidence_report(self, filename: str = "confidence_report.json"):
        """Save detailed confidence report"""
        report = {
            'timestamp': time.time(),
            'system_confidence': self.get_system_confidence(),
            'sensors': {}
        }
        
        for name, sensor in self.sensors.items():
            if sensor.confidence_history:
                latest = sensor.confidence_history[-1]
                report['sensors'][name] = {
                    'latest_metrics': latest.__dict__,
                    'audit_results': sensor.audit_results,
                    'historical_average': sensor.get_historical_reliability()
                }
                
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Confidence report saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Create confidence manager
    manager = SensorConfidenceManager()
    
    # Add sensors
    manager.add_sensor(IMUConfidence())
    manager.add_sensor(CameraConfidence("0"))
    manager.add_sensor(CameraConfidence("1"))
    
    # Simulate some contexts and data
    # Context: Moving with vision active
    manager.update_context(
        motion_detected=True,
        vision_active=True,
        good_lighting=True,
        stereo_vision=True
    )
    
    # Example sensor data
    sensor_data = {
        'IMU': {
            'accel_x': 0.1, 'accel_y': 0.2, 'accel_z': 9.7,
            'gyro_x': 0.5, 'gyro_y': -0.3, 'gyro_z': 0.1
        },
        'Camera_0': {
            'mean_brightness': 128,
            'std_brightness': 45,
            'blur_score': 10
        },
        'Camera_1': {
            'mean_brightness': 130,
            'std_brightness': 48,
            'blur_score': 12
        }
    }
    
    # Update confidence
    results = manager.update_all_confidence(sensor_data)
    
    # Display results
    print("\n=== Sensor Confidence Report ===")
    print(f"System Confidence: {manager.get_system_confidence():.2%}\n")
    
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Raw Confidence: {metrics.raw_confidence:.2%}")
        print(f"  Contextual Relevance: {metrics.contextual_relevance:.2%}")
        print(f"  Historical Reliability: {metrics.historical_reliability:.2%}")
        print(f"  Fractal Confidence: {metrics.fractal_confidence:.2%}")
        print()
    
    # Save report
    manager.save_confidence_report()