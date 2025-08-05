#!/usr/bin/env python3
"""
IMU Vertical Mount Confidence Calibration
Implements confidence-based axis remapping and sensor validation
"""

import numpy as np
import time
from typing import Dict
from confidence_framework import IMUConfidence, ConfidenceMetrics
import json

# Try to import IMU, use mock if not available
try:
    from yahboom_imu import YahboomIMU
except ImportError:
    print("⚠️  YahboomIMU not available, using mock IMU")
    class YahboomIMU:
        """Mock IMU for testing without hardware"""
        def read_data(self):
            return {
                'accel_x': np.random.normal(0, 0.1),
                'accel_y': np.random.normal(0, 0.1),
                'accel_z': np.random.normal(9.81, 0.1),
                'gyro_x': np.random.normal(0, 1.0),
                'gyro_y': np.random.normal(0, 1.0),
                'gyro_z': np.random.normal(0, 1.0)
            }

class VerticalIMUConfidence(IMUConfidence):
    """Extended IMU confidence for vertical mount calibration"""
    
    def __init__(self):
        super().__init__()
        self.imu = YahboomIMU()
        self.axis_confidence = {'x': 0.5, 'y': 0.5, 'z': 0.5}
        self.remapped_axes = None
        
    def audit(self) -> Dict[str, float]:
        """Perform actual IMU audit with vertical mount detection"""
        print(f"🔍 Auditing {self.name} - Vertical Mount Configuration")
        print("=" * 50)
        
        # Collect gravity samples
        print("Collecting gravity vector (keep device still)...")
        gravity = self._measure_gravity()
        
        # Determine which axis has gravity
        gravity_axis = np.argmax(np.abs(gravity))
        gravity_magnitude = np.linalg.norm(gravity)
        
        print(f"\nGravity vector: [{gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f}]")
        print(f"Magnitude: {gravity_magnitude:.2f} g")
        print(f"Primary axis: {['X', 'Y', 'Z'][gravity_axis]}")
        
        # Base confidence scores
        audit_results = {}
        
        # Accelerometer confidence based on gravity reading
        accel_conf = 1.0 - abs(gravity_magnitude - 9.81) / 5.0
        audit_results['accelerometer'] = max(0.2, min(1.0, accel_conf))
        
        # Gyroscope confidence (test for noise/drift)
        gyro_noise = self._measure_gyro_noise()
        audit_results['gyroscope'] = 1.0 - min(gyro_noise / 10.0, 0.8)
        
        # Magnetometer confidence - severely reduced for vertical mount
        if gravity_axis != 2:  # Not standard horizontal mount
            print("\n⚠️  Vertical mount detected!")
            audit_results['magnetometer'] = 0.15  # Very low confidence
            audit_results['heading_reliability'] = 0.1
            
            # But boost confidence for the axes that still work
            working_axes = [i for i in range(3) if i != gravity_axis]
            for i in working_axes:
                axis_name = ['x', 'y', 'z'][i]
                self.axis_confidence[axis_name] = 0.85
            
            # Gravity axis has different characteristics
            self.axis_confidence[['x', 'y', 'z'][gravity_axis]] = 0.6
        else:
            print("\n✅ Standard horizontal mount detected")
            audit_results['magnetometer'] = 0.85
            audit_results['heading_reliability'] = 0.8
            self.axis_confidence = {'x': 0.9, 'y': 0.9, 'z': 0.9}
        
        # Temperature sensor
        audit_results['temperature'] = 0.9
        
        # Per-axis confidence
        for axis, conf in self.axis_confidence.items():
            audit_results[f'axis_{axis}'] = conf
        
        print(f"\n📊 Audit Results:")
        for component, confidence in audit_results.items():
            print(f"  {component}: {confidence:.0%}")
        
        return audit_results
    
    def _measure_gravity(self, duration=2.0):
        """Measure average gravity vector"""
        samples = []
        start = time.time()
        
        while time.time() - start < duration:
            data = self.imu.read_data()
            if data:
                samples.append([
                    data['accel_x'],
                    data['accel_y'], 
                    data['accel_z']
                ])
            time.sleep(0.01)
        
        if samples:
            return np.mean(samples, axis=0)
        return np.array([0, 0, 9.81])  # Default
    
    def _measure_gyro_noise(self, duration=2.0):
        """Measure gyroscope noise when stationary"""
        samples = []
        start = time.time()
        
        while time.time() - start < duration:
            data = self.imu.read_data()
            if data:
                samples.append([
                    data['gyro_x'],
                    data['gyro_y'],
                    data['gyro_z']
                ])
            time.sleep(0.01)
        
        if samples:
            # Return standard deviation as noise measure
            return np.std(samples)
        return 0.0
    
    def compute_contextual_confidence(self, sensor_data: Dict, context: Dict) -> Dict[str, float]:
        """Compute per-component confidence based on context"""
        component_confidence = {}
        
        # Accelerometer confidence
        accel_conf = self.audit_results.get('accelerometer', 0.5)
        if context.get('high_acceleration', False):
            # Might be saturating
            accel_conf *= 0.7
        component_confidence['accelerometer'] = accel_conf
        
        # Gyroscope confidence  
        gyro_conf = self.audit_results.get('gyroscope', 0.5)
        gyro_mag = np.sqrt(
            sensor_data.get('gyro_x', 0)**2 +
            sensor_data.get('gyro_y', 0)**2 +
            sensor_data.get('gyro_z', 0)**2
        )
        if gyro_mag > 300:  # High rotation rate
            gyro_conf *= 0.8  # Might be approaching limits
        component_confidence['gyroscope'] = gyro_conf
        
        # Magnetometer confidence
        mag_conf = self.audit_results.get('magnetometer', 0.15)
        if context.get('near_metal', False):
            mag_conf *= 0.5
        if context.get('need_heading', False) and mag_conf < 0.3:
            print("⚠️  Low magnetometer confidence for heading task!")
        component_confidence['magnetometer'] = mag_conf
        
        return component_confidence
    
    def suggest_remapping(self):
        """Suggest axis remapping based on confidence"""
        print("\n🔄 Suggested Axis Remapping:")
        
        # Find gravity axis (lowest rotational confidence)
        gravity_axis = min(self.axis_confidence.items(), key=lambda x: x[1])[0]
        
        if gravity_axis != 'z':
            print(f"  Gravity is along {gravity_axis.upper()} axis")
            print("  Suggested remapping for standard coordinates:")
            
            if gravity_axis == 'x':
                print("    Camera X (right) -> IMU Y")
                print("    Camera Y (down)  -> IMU Z") 
                print("    Camera Z (forward) -> IMU -X")
                remapping = {'x': 'y', 'y': 'z', 'z': '-x'}
            elif gravity_axis == 'y':
                print("    Camera X (right) -> IMU X")
                print("    Camera Y (down)  -> IMU Z")
                print("    Camera Z (forward) -> IMU Y")  
                remapping = {'x': 'x', 'y': 'z', 'z': 'y'}
            
            self.remapped_axes = remapping
            return remapping
        else:
            print("  Standard mounting detected - no remapping needed")
            return None

def run_vertical_confidence_audit():
    """Run complete confidence audit for vertical IMU"""
    print("🎯 IMU Vertical Mount Confidence Audit")
    print("This will calibrate your IMU and establish confidence metrics\n")
    
    # Create vertical IMU confidence evaluator
    imu_conf = VerticalIMUConfidence()
    
    # Run audit
    audit_results = imu_conf.audit()
    
    # Suggest remapping
    remapping = imu_conf.suggest_remapping()
    
    # Test in different contexts
    print("\n📋 Testing Contextual Confidence:")
    
    contexts = [
        {'name': 'Stationary', 'motion_detected': False, 'vision_active': False},
        {'name': 'Moving', 'motion_detected': True, 'vision_active': True},
        {'name': 'Navigation', 'navigation_active': True, 'need_heading': True},
    ]
    
    for ctx in contexts:
        print(f"\n  Context: {ctx['name']}")
        
        # Get current IMU data
        data = imu_conf.imu.read_data()
        if data:
            sensor_dict = {
                'accel_x': data['accel_x'],
                'accel_y': data['accel_y'],
                'accel_z': data['accel_z'],
                'gyro_x': data['gyro_x'],
                'gyro_y': data['gyro_y'],
                'gyro_z': data['gyro_z']
            }
            
            # Compute component confidence
            comp_conf = imu_conf.compute_contextual_confidence(sensor_dict, ctx)
            
            # Overall confidence
            metrics = imu_conf.update_confidence(sensor_dict, ctx)
            
            print(f"    Fractal Confidence: {metrics.fractal_confidence:.0%}")
            print(f"    Contextual Relevance: {metrics.contextual_relevance:.0%}")
            
            if ctx.get('need_heading') and comp_conf['magnetometer'] < 0.3:
                print(f"    ⚠️  Magnetometer confidence too low for heading!")
    
    # Save configuration
    config = {
        'audit_results': audit_results,
        'axis_confidence': imu_conf.axis_confidence,
        'remapped_axes': remapping,
        'mounting': 'vertical' if remapping else 'horizontal',
        'timestamp': time.time()
    }
    
    with open('imu_confidence_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Configuration saved to imu_confidence_config.json")
    
    print("\n📌 Summary:")
    print(f"  Overall IMU Confidence: {metrics.fractal_confidence:.0%}")
    print(f"  Magnetometer Status: {'❌ Unreliable' if audit_results['magnetometer'] < 0.3 else '✅ Good'}")
    print(f"  Recommended Action: {'Remount horizontally for compass' if remapping else 'Current mount is optimal'}")

if __name__ == "__main__":
    run_vertical_confidence_audit()