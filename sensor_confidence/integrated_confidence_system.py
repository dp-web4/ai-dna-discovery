#!/usr/bin/env python3
"""
Integrated Sensor Confidence System
Combines IMU, camera, and Bluetooth audio confidence into a unified consciousness
Implements fractal attention based on motion context and sensor relevance
"""

import time
import numpy as np
from typing import Dict, List, Optional
from confidence_framework import SensorConfidenceManager, ConfidenceMetrics
from imu_vertical_confidence import VerticalIMUConfidence
from camera_confidence import StereoCameraConfidence
from bluetooth_audio_confidence import BluetoothAudioConfidence
import json
from dataclasses import dataclass

@dataclass
class AttentionState:
    """Current attention state of the system"""
    primary_sensors: List[str]  # Which sensors have high attention
    motion_level: float  # 0-1: How much motion is detected
    task_context: str   # Current primary task
    confidence_threshold: float  # Minimum confidence to trust sensor
    
class IntegratedSensorConsciousness:
    """
    Integrated sensor consciousness system implementing fractal attention
    Each sensor maintains its own confidence, and system attention adapts based on context
    """
    
    def __init__(self):
        self.manager = SensorConfidenceManager()
        self.attention_state = AttentionState(
            primary_sensors=[],
            motion_level=0.0,
            task_context="idle",
            confidence_threshold=0.5
        )
        
        # Sensor instances
        self.imu_confidence = None
        self.camera_confidence = None
        self.audio_confidence = None
        
        # Attention history for temporal patterns
        self.attention_history = []
        
        # Initialize sensors
        self._initialize_sensors()
    
    def _initialize_sensors(self):
        """Initialize all sensor confidence evaluators"""
        print("🧠 Initializing Integrated Sensor Consciousness")
        print("=" * 50)
        
        try:
            # IMU with vertical mount handling
            print("Initializing IMU confidence...")
            self.imu_confidence = VerticalIMUConfidence()
            self.manager.add_sensor(self.imu_confidence)
            
            # Stereo cameras
            print("Initializing camera confidence...")
            self.camera_confidence = StereoCameraConfidence(0, 1)
            
            # Add individual cameras to manager
            self.manager.add_sensor(self.camera_confidence.left_conf)
            self.manager.add_sensor(self.camera_confidence.right_conf)
            
            # Bluetooth audio
            print("Initializing Bluetooth audio confidence...")
            self.audio_confidence = BluetoothAudioConfidence()
            self.manager.add_sensor(self.audio_confidence)
            
            print("✅ All sensors initialized")
            
        except Exception as e:
            print(f"⚠️  Sensor initialization error: {e}")
    
    def update_consciousness(self, motion_detected: bool = False, 
                           task_context: str = "idle") -> Dict:
        """
        Update the entire sensor consciousness system
        Returns comprehensive confidence and attention metrics
        """
        
        # Update motion level (exponential smoothing)
        new_motion = 1.0 if motion_detected else 0.0
        self.attention_state.motion_level = (
            0.7 * self.attention_state.motion_level + 
            0.3 * new_motion
        )
        
        self.attention_state.task_context = task_context
        
        # Gather sensor data
        sensor_data = self._collect_sensor_data()
        
        # Update system context based on motion and task
        context = self._compute_system_context()
        
        # Update all sensor confidences
        confidence_results = self.manager.update_all_confidence(sensor_data, context)
        
        # Compute fractal attention allocation
        attention_allocation = self._compute_attention_allocation(confidence_results)
        
        # Update primary sensors based on attention
        self._update_primary_sensors(attention_allocation)
        
        # Compile consciousness report
        consciousness_report = {
            'timestamp': time.time(),
            'motion_level': self.attention_state.motion_level,
            'task_context': task_context,
            'system_confidence': self.manager.get_system_confidence(),
            'sensor_confidences': {
                name: {
                    'fractal_confidence': metrics.fractal_confidence,
                    'contextual_relevance': metrics.contextual_relevance,
                    'raw_confidence': metrics.raw_confidence
                }
                for name, metrics in confidence_results.items()
            },
            'attention_allocation': attention_allocation,
            'primary_sensors': self.attention_state.primary_sensors,
            'recommendations': self._generate_recommendations(confidence_results)
        }
        
        # Store in history
        self.attention_history.append(consciousness_report)
        if len(self.attention_history) > 100:
            self.attention_history.pop(0)
        
        return consciousness_report
    
    def _collect_sensor_data(self) -> Dict[str, Dict]:
        """Collect current data from all sensors"""
        sensor_data = {}
        
        # IMU data
        if self.imu_confidence and self.imu_confidence.imu:
            try:
                imu_data = self.imu_confidence.imu.read_data()
                if imu_data:
                    sensor_data['IMU'] = {
                        'accel_x': imu_data['accel_x'],
                        'accel_y': imu_data['accel_y'],
                        'accel_z': imu_data['accel_z'],
                        'gyro_x': imu_data['gyro_x'],
                        'gyro_y': imu_data['gyro_y'],
                        'gyro_z': imu_data['gyro_z']
                    }
            except Exception as e:
                print(f"IMU data collection error: {e}")
                sensor_data['IMU'] = {}
        
        # Camera data (simplified - would include image metrics)
        sensor_data['left_eye'] = {
            'mean_brightness': 128,  # Would be computed from actual frames
            'std_brightness': 45,
            'blur_score': 15
        }
        
        sensor_data['right_eye'] = {
            'mean_brightness': 130,
            'std_brightness': 47,
            'blur_score': 12
        }
        
        # Bluetooth audio data (simplified)
        sensor_data['Bluetooth_Audio'] = {
            'connection_quality': 0.8,
            'audio_dropouts': 1,
            'latency_ms': 85
        }
        
        return sensor_data
    
    def _compute_system_context(self) -> Dict:
        """Compute system context based on current state"""
        context = {
            'motion_detected': self.attention_state.motion_level > 0.3,
            'high_motion': self.attention_state.motion_level > 0.7,
            'vision_active': self.attention_state.task_context in ['navigation', 'vision', 'tracking'],
            'audio_output': self.attention_state.task_context in ['communication', 'media'],
            'navigation_active': self.attention_state.task_context == 'navigation',
            'need_heading': self.attention_state.task_context in ['navigation', 'mapping'],
            'stereo_vision': self.attention_state.task_context in ['depth_perception', 'navigation'],
            'mobile': self.attention_state.motion_level > 0.5
        }
        
        return context
    
    def _compute_attention_allocation(self, confidence_results: Dict[str, ConfidenceMetrics]) -> Dict[str, float]:
        """
        Compute fractal attention allocation
        Higher confidence and relevance get more attention
        """
        attention = {}
        total_attention = 0
        
        # Base attention allocation
        for sensor_name, metrics in confidence_results.items():
            # Attention based on confidence * relevance
            sensor_attention = metrics.fractal_confidence * metrics.contextual_relevance
            
            # Boost attention for critical sensors in high motion
            if self.attention_state.motion_level > 0.5 and sensor_name == 'IMU':
                sensor_attention *= 1.5
            
            # Boost attention for vision during visual tasks
            if self.attention_state.task_context in ['vision', 'navigation'] and 'eye' in sensor_name:
                sensor_attention *= 1.3
            
            attention[sensor_name] = sensor_attention
            total_attention += sensor_attention
        
        # Normalize to sum to 1.0
        if total_attention > 0:
            for sensor_name in attention:
                attention[sensor_name] /= total_attention
        
        return attention
    
    def _update_primary_sensors(self, attention_allocation: Dict[str, float]):
        """Update which sensors are currently primary focus"""
        # Sort sensors by attention
        sorted_sensors = sorted(attention_allocation.items(), 
                              key=lambda x: x[1], reverse=True)
        
        # Primary sensors are top 50% of attention or above threshold
        threshold = max(0.2, np.mean(list(attention_allocation.values())))
        
        self.attention_state.primary_sensors = [
            name for name, attention in sorted_sensors 
            if attention >= threshold
        ][:3]  # Max 3 primary sensors
    
    def _generate_recommendations(self, confidence_results: Dict[str, ConfidenceMetrics]) -> List[str]:
        """Generate actionable recommendations based on confidence"""
        recommendations = []
        
        for sensor_name, metrics in confidence_results.items():
            if metrics.fractal_confidence < 0.3:
                if sensor_name == 'IMU':
                    if hasattr(self.imu_confidence, 'audit_results'):
                        mag_conf = self.imu_confidence.audit_results.get('magnetometer', 1.0)
                        if mag_conf < 0.3:
                            recommendations.append("📍 Remount IMU horizontally for compass functionality")
                        else:
                            recommendations.append("🔧 IMU needs recalibration or replacement")
                
                elif 'eye' in sensor_name:
                    recommendations.append(f"📷 {sensor_name.replace('_', ' ').title()} has poor image quality - check lighting/focus")
                
                elif sensor_name == 'Bluetooth_Audio':
                    recommendations.append("🎧 Bluetooth audio unreliable - check connection or use wired audio")
            
            elif metrics.contextual_relevance > 0.8 and metrics.fractal_confidence < 0.6:
                recommendations.append(f"⚠️  {sensor_name} is critical for current task but has medium confidence")
        
        # System-level recommendations
        system_conf = self.manager.get_system_confidence()
        if system_conf < 0.5:
            recommendations.append("🔄 System confidence low - run full sensor audit")
        
        return recommendations
    
    def print_consciousness_state(self, report: Dict):
        """Print human-readable consciousness state"""
        print(f"\n🧠 Sensor Consciousness State")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(f"Motion Level: {report['motion_level']:.0%}")
        print(f"Task Context: {report['task_context']}")
        print(f"System Confidence: {report['system_confidence']:.0%}")
        
        print(f"\n🎯 Primary Sensors: {', '.join(report['primary_sensors'])}")
        
        print(f"\n📊 Sensor Details:")
        for sensor, conf in report['sensor_confidences'].items():
            attention = report['attention_allocation'].get(sensor, 0)
            primary = "🔥" if sensor in report['primary_sensors'] else "  "
            
            print(f"  {primary} {sensor}:")
            print(f"     Confidence: {conf['fractal_confidence']:.0%}")
            print(f"     Relevance:  {conf['contextual_relevance']:.0%}")
            print(f"     Attention:  {attention:.0%}")
        
        if report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"  {rec}")
    
    def run_continuous_consciousness(self, update_interval: float = 5.0):
        """Run continuous consciousness monitoring"""
        print(f"🔄 Starting continuous sensor consciousness monitoring")
        print(f"Update interval: {update_interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Simulate different contexts
                motion = self.attention_state.motion_level > 0.1 or np.random.random() < 0.3
                
                contexts = ['idle', 'vision', 'navigation', 'communication']
                if np.random.random() < 0.1:  # 10% chance to change context
                    context = np.random.choice(contexts)
                else:
                    context = self.attention_state.task_context
                
                # Update consciousness
                report = self.update_consciousness(motion, context)
                
                # Display results
                self.print_consciousness_state(report)
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Consciousness monitoring stopped")
            
            # Save final report
            with open('final_consciousness_report.json', 'w') as f:
                json.dump(self.attention_history[-10:], f, indent=2)
            print("📄 Final report saved to final_consciousness_report.json")

# Example usage
if __name__ == "__main__":
    print("🧠 Integrated Sensor Consciousness System")
    print("This implements fractal attention based on sensor confidence")
    print()
    
    # Create integrated system
    consciousness = IntegratedSensorConsciousness()
    
    # Run a few test cycles
    test_scenarios = [
        ('idle', False),
        ('vision', False),
        ('navigation', True),
        ('communication', False),
    ]
    
    print("🧪 Testing different scenarios:")
    for i, (context, motion) in enumerate(test_scenarios):
        print(f"\n--- Scenario {i+1}: {context.title()} {'with motion' if motion else 'stationary'} ---")
        
        report = consciousness.update_consciousness(motion, context)
        consciousness.print_consciousness_state(report)
        
        time.sleep(2)
    
    print("\n🎯 Key Features Demonstrated:")
    print("  ✅ Fractal confidence (raw + contextual + historical)")
    print("  ✅ Dynamic attention allocation")
    print("  ✅ Context-aware sensor relevance")
    print("  ✅ Actionable recommendations")
    print("  ✅ Vertical IMU mount handling")
    
    print("\n💡 Next Steps:")
    print("  • Run: python3 integrated_confidence_system.py")
    print("  • For continuous monitoring, uncomment consciousness.run_continuous_consciousness()")
    
    # Uncomment to run continuous monitoring:
    # consciousness.run_continuous_consciousness()