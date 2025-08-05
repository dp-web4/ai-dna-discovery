#!/usr/bin/env python3
"""
Bluetooth Audio Confidence Implementation
Evaluates Bluetooth audio input/output confidence and quality
"""

import subprocess
import re
import time
import numpy as np
from confidence_framework import SensorConfidence
from typing import Dict, List, Optional, Tuple
import json

class BluetoothAudioConfidence(SensorConfidence):
    """Confidence evaluation for Bluetooth audio devices"""
    
    def __init__(self):
        super().__init__("Bluetooth_Audio")
        self.connected_devices = {}
        self.signal_strengths = {}
        self.audio_quality_history = []
        
    def audit(self) -> Dict[str, float]:
        """Audit Bluetooth audio devices and connections"""
        print("🔍 Auditing Bluetooth Audio System")
        print("=" * 40)
        
        audit_results = {}
        
        # Check Bluetooth service status
        bt_service = self._check_bluetooth_service()
        audit_results['bluetooth_service'] = bt_service
        
        if bt_service < 0.5:
            print("❌ Bluetooth service not running properly")
            return audit_results
        
        # Scan for devices
        devices = self._scan_audio_devices()
        audit_results['device_discovery'] = 1.0 if devices else 0.0
        
        # Test each connected audio device
        device_scores = {}
        for device_id, device_info in devices.items():
            score = self._audit_device(device_id, device_info)
            device_scores[device_id] = score
            
        audit_results['devices'] = device_scores
        
        # Overall connectivity
        if device_scores:
            audit_results['overall_connectivity'] = np.mean(list(device_scores.values()))
        else:
            audit_results['overall_connectivity'] = 0.0
        
        # Audio routing capability
        routing_quality = self._test_audio_routing()
        audit_results['audio_routing'] = routing_quality
        
        print(f"\nBluetooth service: {bt_service:.0%}")
        print(f"Connected devices: {len(device_scores)}")
        print(f"Overall connectivity: {audit_results['overall_connectivity']:.0%}")
        
        return audit_results
    
    def _check_bluetooth_service(self) -> float:
        """Check if Bluetooth service is running"""
        try:
            # Check bluetoothd service
            result = subprocess.run(['systemctl', 'is-active', 'bluetooth'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and 'active' in result.stdout:
                return 1.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error checking Bluetooth service: {e}")
            return 0.0
    
    def _scan_audio_devices(self) -> Dict[str, Dict]:
        """Scan for connected Bluetooth audio devices"""
        devices = {}
        
        try:
            # Use bluetoothctl to list all devices and filter connected ones
            result = subprocess.run(['bluetoothctl', 'devices'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        # Parse device info: "Device AA:BB:CC:DD:EE:FF Device Name"
                        match = re.match(r'Device\s+([A-F0-9:]{17})\s+(.+)', line)
                        if match:
                            mac_address = match.group(1)
                            device_name = match.group(2)
                            
                            # Check if it's connected and an audio device
                            if self._is_connected(mac_address) and self._is_audio_device(mac_address):
                                devices[mac_address] = {
                                    'name': device_name,
                                    'mac': mac_address,
                                    'type': 'audio'
                                }
            
        except Exception as e:
            print(f"Error scanning devices: {e}")
        
        return devices
    
    def _is_connected(self, mac_address: str) -> bool:
        """Check if device is currently connected"""
        try:
            result = subprocess.run(['bluetoothctl', 'info', mac_address], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return 'Connected: yes' in result.stdout
                
        except Exception as e:
            print(f"Error checking connection for {mac_address}: {e}")
        
        return False
    
    def _is_audio_device(self, mac_address: str) -> bool:
        """Check if device supports audio profiles"""
        try:
            # Check device info for audio services
            result = subprocess.run(['bluetoothctl', 'info', mac_address], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                output = result.stdout.lower()
                # Look for audio-related UUIDs/services
                audio_indicators = [
                    'a2dp',  # Advanced Audio Distribution Profile
                    'avrcp', # Audio/Video Remote Control Profile
                    'headset',
                    'handsfree',
                    'audio'
                ]
                
                return any(indicator in output for indicator in audio_indicators)
                
        except Exception as e:
            print(f"Error checking device {mac_address}: {e}")
        
        return False
    
    def _audit_device(self, device_id: str, device_info: Dict) -> float:
        """Audit individual Bluetooth audio device"""
        print(f"\n  🎧 Testing {device_info['name']}")
        
        scores = []
        
        # Connection stability
        connection_score = self._test_connection_stability(device_id)
        scores.append(connection_score)
        print(f"    Connection stability: {connection_score:.0%}")
        
        # Signal strength/quality
        signal_score = self._test_signal_strength(device_id)
        scores.append(signal_score)
        print(f"    Signal strength: {signal_score:.0%}")
        
        # Audio latency
        latency_score = self._test_audio_latency(device_id)
        scores.append(latency_score)
        print(f"    Latency: {latency_score:.0%}")
        
        # Battery level (if available)
        battery_score = self._check_battery_level(device_id)
        if battery_score >= 0:
            scores.append(battery_score)
            print(f"    Battery: {battery_score:.0%}")
        
        return np.mean(scores) if scores else 0.0
    
    def _test_connection_stability(self, device_id: str) -> float:
        """Test connection stability over time"""
        stable_connections = 0
        total_tests = 5
        
        for _ in range(total_tests):
            try:
                # Check if device is still connected
                result = subprocess.run(['bluetoothctl', 'info', device_id], 
                                      capture_output=True, text=True, timeout=3)
                
                if result.returncode == 0 and 'connected: yes' in result.stdout.lower():
                    stable_connections += 1
                    
                time.sleep(1)
                
            except Exception:
                pass
        
        return stable_connections / total_tests
    
    def _test_signal_strength(self, device_id: str) -> float:
        """Test Bluetooth signal strength/quality"""
        try:
            # Try to get RSSI (Received Signal Strength Indicator)
            result = subprocess.run(['bluetoothctl', 'info', device_id], 
                                  capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                # Look for RSSI in the output
                rssi_match = re.search(r'RSSI:\s*(-?\d+)', result.stdout)
                if rssi_match:
                    rssi = int(rssi_match.group(1))
                    # Convert RSSI to confidence score (typical range: -30 to -90 dBm)
                    # -30 dBm = excellent, -90 dBm = poor
                    confidence = max(0, min(1, (rssi + 90) / 60))
                    return confidence
            
            # If no RSSI available, use connection success as proxy
            return 0.7  # Assume decent signal if connected
            
        except Exception:
            return 0.5
    
    def _test_audio_latency(self, device_id: str) -> float:
        """Test audio latency (simplified)"""
        # This is complex to measure accurately without specialized tools
        # For now, return score based on device type/codec
        
        try:
            result = subprocess.run(['bluetoothctl', 'info', device_id], 
                                  capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                output = result.stdout.lower()
                
                # Check for low-latency codecs
                if 'aptx' in output or 'ldac' in output:
                    return 0.9  # Excellent latency
                elif 'sbc' in output:
                    return 0.6  # Standard latency
                else:
                    return 0.7  # Unknown, assume decent
                    
        except Exception:
            pass
        
        return 0.5
    
    def _check_battery_level(self, device_id: str) -> float:
        """Check device battery level if available"""
        try:
            result = subprocess.run(['bluetoothctl', 'info', device_id], 
                                  capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                # Look for battery percentage
                battery_match = re.search(r'Battery Percentage:\s*(\d+)', result.stdout)
                if battery_match:
                    battery_level = int(battery_match.group(1))
                    # Convert to confidence (low battery = unreliable)
                    if battery_level > 50:
                        return 1.0
                    elif battery_level > 20:
                        return 0.7
                    else:
                        return 0.3
            
            return -1  # No battery info available
            
        except Exception:
            return -1
    
    def _test_audio_routing(self) -> float:
        """Test audio routing capabilities"""
        try:
            # Check PulseAudio/PipeWire status
            result = subprocess.run(['pactl', 'info'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Check for Bluetooth sinks
                sink_result = subprocess.run(['pactl', 'list', 'short', 'sinks'], 
                                           capture_output=True, text=True, timeout=5)
                
                if sink_result.returncode == 0:
                    bluetooth_sinks = [line for line in sink_result.stdout.split('\n') 
                                     if 'bluetooth' in line.lower()]
                    
                    return 1.0 if bluetooth_sinks else 0.5
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def compute_raw_confidence(self, sensor_data: Dict) -> float:
        """Compute confidence from current Bluetooth audio state"""
        if not sensor_data:
            return 0.0
        
        confidences = []
        
        # Connection quality
        connection_quality = sensor_data.get('connection_quality', 0.5)
        confidences.append(connection_quality)
        
        # Audio quality metrics
        audio_dropouts = sensor_data.get('audio_dropouts', 0)
        dropout_conf = max(0, 1.0 - audio_dropouts / 10.0)  # 10+ dropouts = 0 confidence
        confidences.append(dropout_conf)
        
        # Latency
        latency_ms = sensor_data.get('latency_ms', 100)
        latency_conf = max(0, 1.0 - max(0, latency_ms - 50) / 200.0)  # 50ms good, 250ms+ bad
        confidences.append(latency_conf)
        
        return np.mean(confidences)
    
    def evaluate_context(self, context: Dict) -> float:
        """Evaluate Bluetooth audio relevance in current context"""
        relevance = 0.1  # Base relevance
        
        # Audio output needed
        if context.get('audio_output', False):
            relevance += 0.6
        
        # Voice input needed
        if context.get('voice_input', False):
            relevance += 0.5
        
        # Communication active
        if context.get('communication', False):
            relevance += 0.4
        
        # Mobility context (Bluetooth more relevant when moving)
        if context.get('mobile', False):
            relevance += 0.2
        
        # Wired audio available (reduces Bluetooth relevance)
        if context.get('wired_audio_available', False):
            relevance *= 0.7
        
        return min(relevance, 1.0)

class BluetoothAudioMonitor:
    """Continuous monitoring of Bluetooth audio confidence"""
    
    def __init__(self):
        self.confidence_evaluator = BluetoothAudioConfidence()
        self.monitoring = False
        
    def start_monitoring(self, interval: float = 30.0):
        """Start continuous confidence monitoring"""
        print(f"🔄 Starting Bluetooth audio monitoring (every {interval}s)")
        
        self.monitoring = True
        while self.monitoring:
            try:
                # Run audit
                audit_results = self.confidence_evaluator.audit()
                
                # Context example (would come from system state)
                context = {
                    'audio_output': True,
                    'voice_input': False,
                    'mobile': False
                }
                
                # Sensor data example (would come from audio system)
                sensor_data = {
                    'connection_quality': audit_results.get('overall_connectivity', 0.0),
                    'audio_dropouts': 0,  # Would be measured
                    'latency_ms': 80      # Would be measured
                }
                
                # Update confidence
                metrics = self.confidence_evaluator.update_confidence(sensor_data, context)
                
                print(f"\n🎵 Bluetooth Audio Confidence: {metrics.fractal_confidence:.0%}")
                print(f"   Contextual Relevance: {metrics.contextual_relevance:.0%}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped")
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False

# Example usage
if __name__ == "__main__":
    print("🎧 Bluetooth Audio Confidence Test")
    
    confidence = BluetoothAudioConfidence()
    audit_results = confidence.audit()
    
    print("\n📊 Audit Results:")
    for component, score in audit_results.items():
        if isinstance(score, dict):
            print(f"  {component}:")
            for sub_comp, sub_score in score.items():
                print(f"    {sub_comp}: {sub_score:.0%}")
        else:
            print(f"  {component}: {score:.0%}")
    
    # Test contextual confidence
    test_contexts = [
        {'name': 'Music Playback', 'audio_output': True, 'mobile': False},
        {'name': 'Voice Call', 'audio_output': True, 'voice_input': True, 'communication': True},
        {'name': 'Mobile Gaming', 'audio_output': True, 'mobile': True},
    ]
    
    print("\n🎯 Contextual Relevance Tests:")
    for ctx in test_contexts:
        relevance = confidence.evaluate_context(ctx)
        print(f"  {ctx['name']}: {relevance:.0%}")
    
    print("\n💡 Recommendations:")
    overall = audit_results.get('overall_connectivity', 0)
    if overall > 0.8:
        print("  ✅ Bluetooth audio system is reliable")
    elif overall > 0.5:
        print("  ⚠️  Bluetooth audio has some issues - monitor closely")
    else:
        print("  ❌ Bluetooth audio unreliable - consider wired alternatives")