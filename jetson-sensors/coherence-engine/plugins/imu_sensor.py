"""
IMU Sensor Plugin for Coherence Engine
Integrates real IMU over serial connection
August 12, 2025
"""

import serial
import numpy as np
import time
from typing import Dict, Any, Optional
import threading
import queue
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.base import SensorBase

class IMUSensorPlugin(SensorBase):
    """Real IMU sensor plugin via serial connection"""
    
    def __init__(self, identity: str = "imu_sensor"):
        super().__init__(identity)
        self.serial_port = None
        self.port_name = "/dev/ttyUSB0"
        self.baud_rate = 115200
        self.latest_data = {
            "acceleration": [0.0, 0.0, 0.0],
            "gyroscope": [0.0, 0.0, 0.0],
            "magnetometer": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
            "temperature": 0.0
        }
        self.data_queue = queue.Queue(maxsize=10)
        self.read_thread = None
        self.running = False
        
    def initialize(self, config: Dict[str, Any]):
        """Initialize serial connection to IMU"""
        self.port_name = config.get("port", self.port_name)
        self.baud_rate = config.get("baud_rate", self.baud_rate)
        
        print(f"Initializing IMU sensor on {self.port_name} @ {self.baud_rate}")
        
        try:
            self.serial_port = serial.Serial(
                port=self.port_name,
                baudrate=self.baud_rate,
                timeout=0.1
            )
            
            print(f"✓ IMU serial connection established")
            
            # Start read thread
            self.running = True
            self.read_thread = threading.Thread(target=self._read_loop)
            self.read_thread.daemon = True
            self.read_thread.start()
            
        except serial.SerialException as e:
            print(f"✗ Failed to initialize IMU: {e}")
            # Fall back to simulated data
            self.serial_port = None
            self.running = True
            self.read_thread = threading.Thread(target=self._simulate_loop)
            self.read_thread.daemon = True
            self.read_thread.start()
            
    def teardown(self):
        """Clean up serial connection"""
        self.running = False
        
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
            
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            
        print("IMU sensor shutdown complete")
        
    def _read_loop(self):
        """Background thread for reading IMU data"""
        while self.running and self.serial_port:
            try:
                if self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    
                    # Parse IMU data (assuming JSON format)
                    try:
                        data = json.loads(line)
                        self._update_data(data)
                    except json.JSONDecodeError:
                        # Try simple CSV format: ax,ay,az,gx,gy,gz,mx,my,mz
                        parts = line.split(',')
                        if len(parts) >= 9:
                            self.latest_data["acceleration"] = [float(parts[0]), float(parts[1]), float(parts[2])]
                            self.latest_data["gyroscope"] = [float(parts[3]), float(parts[4]), float(parts[5])]
                            self.latest_data["magnetometer"] = [float(parts[6]), float(parts[7]), float(parts[8])]
                            
            except Exception as e:
                print(f"IMU read error: {e}")
                
            time.sleep(0.01)  # 100Hz update rate
            
    def _simulate_loop(self):
        """Simulate IMU data when hardware not available"""
        while self.running:
            # Generate realistic IMU data
            t = time.time()
            
            # Simulate gentle rotation
            self.latest_data["gyroscope"] = [
                0.1 * np.sin(t * 0.5),
                0.05 * np.cos(t * 0.3),
                0.02 * np.sin(t * 0.7)
            ]
            
            # Simulate gravity + small movements
            self.latest_data["acceleration"] = [
                0.05 * np.sin(t * 2),
                0.05 * np.cos(t * 1.5),
                9.81 + 0.1 * np.sin(t)
            ]
            
            # Simulate magnetic field
            self.latest_data["magnetometer"] = [
                30 + 5 * np.sin(t * 0.1),
                -10 + 3 * np.cos(t * 0.15),
                45
            ]
            
            # Calculate orientation from gyro (simplified)
            self.latest_data["orientation"] = [
                np.degrees(np.arctan2(self.latest_data["acceleration"][1], 
                                     self.latest_data["acceleration"][2])),
                np.degrees(np.arctan2(self.latest_data["acceleration"][0], 
                                     self.latest_data["acceleration"][2])),
                np.degrees(np.arctan2(self.latest_data["magnetometer"][1], 
                                     self.latest_data["magnetometer"][0]))
            ]
            
            self.latest_data["temperature"] = 25 + 5 * np.sin(t * 0.01)
            
            time.sleep(0.01)  # 100Hz update rate
            
    def _update_data(self, data: Dict[str, Any]):
        """Update latest IMU data from parsed input"""
        if "acceleration" in data:
            self.latest_data["acceleration"] = data["acceleration"]
        if "gyroscope" in data:
            self.latest_data["gyroscope"] = data["gyroscope"]
        if "magnetometer" in data:
            self.latest_data["magnetometer"] = data["magnetometer"]
        if "orientation" in data:
            self.latest_data["orientation"] = data["orientation"]
        if "temperature" in data:
            self.latest_data["temperature"] = data["temperature"]
            
    def read(self) -> Dict[str, Any]:
        """Read current IMU data"""
        data = self.latest_data.copy()
        
        # Calculate motion intensity
        accel_mag = np.linalg.norm(data["acceleration"])
        gyro_mag = np.linalg.norm(data["gyroscope"])
        
        # Detect if stationary (low motion)
        stationary = accel_mag < 10.0 and gyro_mag < 0.1
        
        # Detect sudden motion (high acceleration or rotation)
        sudden_motion = accel_mag > 15.0 or gyro_mag > 2.0
        
        # Calculate stability metric (0-1, higher is more stable)
        stability = 1.0 / (1.0 + gyro_mag * 10)
        
        return {
            **data,
            "stationary": stationary,
            "sudden_motion": sudden_motion,
            "stability": stability,
            "confidence": 1.0 if self.serial_port else 0.5,  # Lower confidence for simulated
            "timestamp": time.time()
        }
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Declare IMU sensor capabilities"""
        return {
            "type": "motion",
            "subtype": "9dof_imu",
            "connection": "serial" if self.serial_port else "simulated",
            "update_rate": 100,  # Hz
            "features": [
                "3-axis-acceleration",
                "3-axis-gyroscope", 
                "3-axis-magnetometer",
                "orientation",
                "temperature"
            ],
            "metrics": ["stability", "stationary", "sudden_motion"],
            "confidence_range": [0.0, 1.0]
        }