#!/usr/bin/env python3
"""
IMU Sensor Integration for Jetson Orin Nano
Yahboom CMP10A 10-DOF sensor with proper orientation
"""

import serial
import struct
import time
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from datetime import datetime

@dataclass
class IMUData:
    """IMU sensor data with proper orientation"""
    timestamp: float
    accelerometer: Tuple[float, float, float]  # x, y, z in m/s^2
    gyroscope: Tuple[float, float, float]      # x, y, z in rad/s
    magnetometer: Tuple[float, float, float]   # x, y, z in uT
    euler: Tuple[float, float, float]          # roll, pitch, yaw in degrees
    quaternion: Tuple[float, float, float, float]  # w, x, y, z
    temperature: float  # Celsius
    confidence: float   # 0.0 to 1.0

class IMUSensor:
    """IMU sensor with Web4-aligned confidence and sensor fusion"""
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.is_connected = False
        self.calibration_data = None
        self.confidence_history = []
        self.last_data = None
        
        # Sensor fusion weights
        self.sensor_weights = {
            'accelerometer': 0.3,
            'gyroscope': 0.5,
            'magnetometer': 0.2
        }
        
        # Orientation correction for horizontal mounting
        self.orientation_matrix = np.array([
            [1, 0, 0],   # X axis unchanged
            [0, 1, 0],   # Y axis unchanged  
            [0, 0, 1]    # Z axis unchanged (now properly oriented)
        ])
        
    def connect(self) -> bool:
        """Connect to IMU sensor"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.is_connected = True
            print(f"Connected to IMU at {self.port} ({self.baudrate} baud)")
            return True
        except Exception as e:
            print(f"Failed to connect to IMU: {e}")
            return False
    
    def read_raw_data(self) -> Optional[Dict]:
        """Read raw data from IMU"""
        if not self.is_connected:
            return None
            
        try:
            # Read IMU packet (adjust protocol as needed)
            if self.serial.in_waiting >= 44:  # Typical packet size
                data = self.serial.read(44)
                
                # Parse packet (adjust format based on actual protocol)
                # This is a simplified example
                values = struct.unpack('<11f', data)
                
                return {
                    'accel': values[0:3],
                    'gyro': values[3:6],
                    'mag': values[6:9],
                    'temp': values[9],
                    'timestamp': time.time()
                }
        except Exception as e:
            print(f"Error reading IMU data: {e}")
            return None
    
    def apply_orientation_correction(self, vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply orientation matrix to correct for mounting"""
        corrected = np.dot(self.orientation_matrix, vector)
        return tuple(corrected)
    
    def calculate_euler_angles(self, accel: Tuple, mag: Tuple) -> Tuple[float, float, float]:
        """Calculate euler angles from accelerometer and magnetometer"""
        ax, ay, az = accel
        mx, my, mz = mag
        
        # Calculate roll and pitch from accelerometer
        roll = np.arctan2(ay, az) * 180 / np.pi
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi
        
        # Calculate yaw from magnetometer (compensated for tilt)
        cos_roll = np.cos(roll * np.pi / 180)
        sin_roll = np.sin(roll * np.pi / 180)
        cos_pitch = np.cos(pitch * np.pi / 180)
        sin_pitch = np.sin(pitch * np.pi / 180)
        
        Mx = mx * cos_pitch + my * sin_roll * sin_pitch + mz * cos_roll * sin_pitch
        My = my * cos_roll - mz * sin_roll
        
        yaw = np.arctan2(-My, Mx) * 180 / np.pi
        
        return (roll, pitch, yaw)
    
    def calculate_quaternion(self, euler: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """Convert euler angles to quaternion"""
        roll, pitch, yaw = [angle * np.pi / 180 for angle in euler]
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return (w, x, y, z)
    
    def calculate_confidence(self, data: Dict) -> float:
        """Calculate sensor confidence using Web4 principles"""
        confidence_factors = []
        
        # Check accelerometer magnitude (should be ~9.8 m/s^2 at rest)
        accel_mag = np.linalg.norm(data['accel'])
        accel_confidence = max(0, 1 - abs(accel_mag - 9.8) / 9.8)
        confidence_factors.append(accel_confidence * self.sensor_weights['accelerometer'])
        
        # Check gyroscope stability (lower values = more stable)
        gyro_mag = np.linalg.norm(data['gyro'])
        gyro_confidence = max(0, 1 - min(gyro_mag / 10, 1))  # Normalize to 0-1
        confidence_factors.append(gyro_confidence * self.sensor_weights['gyroscope'])
        
        # Check magnetometer consistency
        mag_mag = np.linalg.norm(data['mag'])
        if mag_mag > 0:
            mag_confidence = min(1, mag_mag / 50)  # Typical Earth field ~25-65 uT
            confidence_factors.append(mag_confidence * self.sensor_weights['magnetometer'])
        
        # Calculate weighted confidence
        raw_confidence = sum(confidence_factors)
        
        # Apply temporal smoothing
        self.confidence_history.append(raw_confidence)
        if len(self.confidence_history) > 10:
            self.confidence_history.pop(0)
        
        return np.mean(self.confidence_history)
    
    def read(self) -> Optional[IMUData]:
        """Read and process IMU data with confidence scoring"""
        raw_data = self.read_raw_data()
        if not raw_data:
            return None
        
        # Apply orientation corrections
        accel = self.apply_orientation_correction(raw_data['accel'])
        gyro = self.apply_orientation_correction(raw_data['gyro'])
        mag = self.apply_orientation_correction(raw_data['mag'])
        
        # Calculate derived values
        euler = self.calculate_euler_angles(accel, mag)
        quaternion = self.calculate_quaternion(euler)
        
        # Calculate confidence
        confidence = self.calculate_confidence({
            'accel': accel,
            'gyro': gyro,
            'mag': mag
        })
        
        # Create IMU data object
        imu_data = IMUData(
            timestamp=raw_data['timestamp'],
            accelerometer=accel,
            gyroscope=gyro,
            magnetometer=mag,
            euler=euler,
            quaternion=quaternion,
            temperature=raw_data.get('temp', 25.0),
            confidence=confidence
        )
        
        self.last_data = imu_data
        return imu_data
    
    def get_sensor_fusion_data(self) -> Dict:
        """Get data formatted for sensor fusion system"""
        if not self.last_data:
            return None
            
        return {
            'type': 'imu',
            'timestamp': self.last_data.timestamp,
            'data': {
                'orientation': self.last_data.euler,
                'angular_velocity': self.last_data.gyroscope,
                'linear_acceleration': self.last_data.accelerometer,
                'magnetic_field': self.last_data.magnetometer,
                'quaternion': self.last_data.quaternion
            },
            'confidence': self.last_data.confidence,
            'metadata': {
                'temperature': self.last_data.temperature,
                'sensor': 'Yahboom CMP10A',
                'mounting': 'horizontal'
            }
        }
    
    def calibrate(self, samples=100):
        """Calibrate IMU sensor"""
        print(f"Calibrating IMU... Keep sensor stationary for {samples/10} seconds")
        
        accel_samples = []
        gyro_samples = []
        mag_samples = []
        
        for i in range(samples):
            raw_data = self.read_raw_data()
            if raw_data:
                accel_samples.append(raw_data['accel'])
                gyro_samples.append(raw_data['gyro'])
                mag_samples.append(raw_data['mag'])
            time.sleep(0.1)
            
            if i % 10 == 0:
                print(f"Calibration progress: {i}/{samples}")
        
        # Calculate offsets
        self.calibration_data = {
            'accel_offset': np.mean(accel_samples, axis=0),
            'gyro_offset': np.mean(gyro_samples, axis=0),
            'mag_offset': np.mean(mag_samples, axis=0),
            'timestamp': datetime.now().isoformat()
        }
        
        print("Calibration complete!")
        return self.calibration_data
    
    def save_calibration(self, filename='imu_calibration.json'):
        """Save calibration data to file"""
        if self.calibration_data:
            with open(filename, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                cal_data = {
                    'accel_offset': self.calibration_data['accel_offset'].tolist(),
                    'gyro_offset': self.calibration_data['gyro_offset'].tolist(),
                    'mag_offset': self.calibration_data['mag_offset'].tolist(),
                    'timestamp': self.calibration_data['timestamp']
                }
                json.dump(cal_data, f, indent=2)
            print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename='imu_calibration.json'):
        """Load calibration data from file"""
        try:
            with open(filename, 'r') as f:
                cal_data = json.load(f)
                self.calibration_data = {
                    'accel_offset': np.array(cal_data['accel_offset']),
                    'gyro_offset': np.array(cal_data['gyro_offset']),
                    'mag_offset': np.array(cal_data['mag_offset']),
                    'timestamp': cal_data['timestamp']
                }
            print(f"Calibration loaded from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False
    
    def close(self):
        """Close IMU connection"""
        if self.serial:
            self.serial.close()
            self.is_connected = False
            print("IMU connection closed")

if __name__ == "__main__":
    # Test IMU sensor
    imu = IMUSensor()
    
    if imu.connect():
        # Try to load existing calibration
        if not imu.load_calibration():
            # Calibrate if no calibration exists
            imu.calibrate()
            imu.save_calibration()
        
        print("\nReading IMU data...")
        print("-" * 60)
        
        try:
            for _ in range(100):
                data = imu.read()
                if data:
                    print(f"Time: {data.timestamp:.2f}")
                    print(f"Euler (R,P,Y): {data.euler[0]:.1f}°, {data.euler[1]:.1f}°, {data.euler[2]:.1f}°")
                    print(f"Confidence: {data.confidence:.2%}")
                    print("-" * 60)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            imu.close()