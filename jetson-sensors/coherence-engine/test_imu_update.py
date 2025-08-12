#!/usr/bin/env python3
"""
Test IMU update to debug why values are static
"""

import time
import numpy as np

class TestIMU:
    def __init__(self):
        self.imu_data = {
            "acceleration": [0.0, 0.0, 9.81],
            "gyroscope": [0.0, 0.0, 0.0],
        }
        
    def simulate_imu(self):
        """Simulate IMU data - updates imu_data dict"""
        t = time.time()
        
        # Update IMU data with simulated values
        self.imu_data["acceleration"] = [
            0.05 * np.sin(t * 2),           # X
            0.05 * np.cos(t * 1.5),         # Y  
            9.81 + 0.1 * np.sin(t)          # Z (gravity + variation)
        ]
        
        self.imu_data["gyroscope"] = [
            0.1 * np.sin(t * 0.5),           # Roll rate
            0.05 * np.cos(t * 0.3),          # Pitch rate
            0.02 * np.sin(t * 0.7)           # Yaw rate
        ]
        
        print(f"t={t:.2f}: acc[0]={self.imu_data['acceleration'][0]:+.4f}, "
              f"gyro[0]={self.imu_data['gyroscope'][0]:+.4f}")

# Test it
test = TestIMU()
print("Initial:", test.imu_data["acceleration"][0], test.imu_data["gyroscope"][0])

for i in range(5):
    test.simulate_imu()
    time.sleep(0.2)
    
print("Final:", test.imu_data["acceleration"][0], test.imu_data["gyroscope"][0])