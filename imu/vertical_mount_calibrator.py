#!/usr/bin/env python3
"""
Vertical Mount Calibrator for IMU
Helps determine correct axis mapping when IMU is mounted vertically
"""

import time
import numpy as np
from yahboom_imu import YahboomIMU
import json

class VerticalMountCalibrator:
    def __init__(self):
        self.imu = YahboomIMU()
        self.samples = []
        
    def collect_gravity_vector(self, duration=3.0):
        """Collect accelerometer data to find gravity direction"""
        print(f"Collecting gravity vector for {duration} seconds...")
        print("Keep the device still!")
        
        start = time.time()
        accel_sum = np.zeros(3)
        count = 0
        
        while time.time() - start < duration:
            data = self.imu.read_data()
            if data:
                accel = np.array([data['accel_x'], data['accel_y'], data['accel_z']])
                accel_sum += accel
                count += 1
                time.sleep(0.01)
        
        if count > 0:
            gravity = accel_sum / count
            gravity_magnitude = np.linalg.norm(gravity)
            gravity_normalized = gravity / gravity_magnitude
            
            print(f"\nGravity vector (raw): {gravity}")
            print(f"Magnitude: {gravity_magnitude:.2f} g")
            print(f"Normalized: {gravity_normalized}")
            
            # Determine which axis gravity is aligned with
            max_axis = np.argmax(np.abs(gravity_normalized))
            axis_names = ['X', 'Y', 'Z']
            
            print(f"\nGravity is primarily along IMU {axis_names[max_axis]} axis")
            print(f"Value: {gravity[max_axis]:.2f} g")
            
            return gravity_normalized, max_axis
        
        return None, None
    
    def test_rotation(self, instruction, axis_name):
        """Test rotation around a specific axis"""
        print(f"\n{instruction}")
        input("Press Enter when ready...")
        
        print("Recording motion for 5 seconds...")
        start = time.time()
        gyro_data = []
        
        while time.time() - start < 5.0:
            data = self.imu.read_data()
            if data:
                gyro = [data['gyro_x'], data['gyro_y'], data['gyro_z']]
                gyro_data.append(gyro)
                time.sleep(0.01)
        
        # Find which axis had the most rotation
        gyro_array = np.array(gyro_data)
        gyro_std = np.std(gyro_array, axis=0)
        max_rotation_axis = np.argmax(gyro_std)
        
        axis_names = ['X', 'Y', 'Z']
        print(f"Maximum rotation detected on IMU {axis_names[max_rotation_axis]} axis")
        print(f"Rotation magnitudes - X: {gyro_std[0]:.1f}, Y: {gyro_std[1]:.1f}, Z: {gyro_std[2]:.1f}")
        
        return max_rotation_axis
    
    def generate_config(self):
        """Generate axis mapping configuration"""
        print("=== IMU Vertical Mount Calibration ===\n")
        
        # Step 1: Find gravity direction
        gravity_vec, gravity_axis = self.collect_gravity_vector()
        if gravity_vec is None:
            print("Failed to detect gravity!")
            return
        
        # Step 2: Test rotations to map axes
        print("\n--- Rotation Tests ---")
        print("We'll determine how IMU axes map to camera coordinates")
        
        # Test yaw (rotation around vertical/gravity)
        yaw_axis = self.test_rotation(
            "ROTATE the device horizontally (like turning your head left/right)",
            "Yaw (vertical)"
        )
        
        # Test pitch 
        pitch_axis = self.test_rotation(
            "TILT the device forward/backward (like nodding)",
            "Pitch"
        )
        
        # Test roll
        roll_axis = self.test_rotation(
            "TILT the device left/right (like tilting your head to shoulder)",
            "Roll"
        )
        
        # Generate mapping
        axis_names = ['x', 'y', 'z']
        camera_to_imu = {}
        
        # Standard camera convention:
        # X = right, Y = down, Z = forward
        # Roll = rotation around Z
        # Pitch = rotation around X  
        # Yaw = rotation around Y
        
        print("\n--- Axis Mapping ---")
        
        # Map based on which IMU axis corresponds to each rotation
        if yaw_axis == gravity_axis:
            # Gravity axis is the yaw axis (vertical)
            camera_to_imu['y'] = axis_names[gravity_axis]
            
            # Determine forward direction (roll axis)
            camera_to_imu['z'] = axis_names[roll_axis]
            
            # Right direction is what's left
            remaining = set([0, 1, 2]) - {gravity_axis, roll_axis}
            camera_to_imu['x'] = axis_names[list(remaining)[0]]
        
        # Check if axes need sign flips
        flip_config = {
            'flip_roll': gravity_vec[gravity_axis] < 0,  # Gravity should be negative
            'flip_pitch': False,
            'flip_yaw': False
        }
        
        config = {
            'axis_map': camera_to_imu,
            'roll_offset': 0.0,
            'pitch_offset': 90.0 if gravity_axis != 2 else 0.0,
            'yaw_offset': 0.0,
            **flip_config,
            'gravity_axis': axis_names[gravity_axis],
            'mounting': 'vertical',
            'magnetometer_warning': 'Vertical mount limits magnetometer accuracy'
        }
        
        print(f"\nGenerated configuration:")
        print(json.dumps(config, indent=2))
        
        # Save configuration
        with open('imu_vertical_calibration.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("\nSaved to imu_vertical_calibration.json")
        
        return config

def main():
    calibrator = VerticalMountCalibrator()
    
    print("This tool will help calibrate your vertically-mounted IMU")
    print("Make sure the IMU is mounted in its final position\n")
    
    try:
        config = calibrator.generate_config()
        
        print("\n=== Recommendations ===")
        print("1. For best magnetometer performance, mount IMU horizontally")
        print("2. Current vertical mount will have limited heading accuracy")
        print("3. Use the generated config with imu_orientation_mapper.py")
        
    except KeyboardInterrupt:
        print("\nCalibration cancelled")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()