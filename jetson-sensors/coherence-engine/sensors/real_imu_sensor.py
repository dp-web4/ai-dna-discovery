"""
Real IMU sensor adapter for serial IMU on Jetson.
Bridges hardware IMU at /dev/ttyUSB0 to coherence engine Protocol.
"""

import serial
import json
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging
import numpy as np

logger = logging.getLogger("real_imu")

@dataclass
class RealIMUSensor:
    """
    Adapts serial IMU device to coherence engine sensor Protocol.
    Returns confidence/activity score based on motion magnitude.
    """
    id: str = "imu"
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 0.1
    
    # Motion detection parameters
    motion_scale: float = 0.1  # Scale motion to [0,1] range
    gravity_filter: float = 0.98  # Complementary filter for gravity removal
    
    # Internal state
    serial_conn: Optional[serial.Serial] = field(default=None, init=False)
    reader_thread: Optional[threading.Thread] = field(default=None, init=False)
    data_queue: queue.Queue = field(default_factory=queue.Queue, init=False)
    latest_reading: Dict[str, float] = field(default_factory=dict, init=False)
    initialized: bool = field(default=False, init=False)
    running: bool = field(default=False, init=False)
    
    # Gravity estimation for motion detection
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 9.8]), init=False)
    
    def __post_init__(self):
        """Initialize serial connection to IMU."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            # Start reader thread
            self.running = True
            self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.reader_thread.start()
            
            # Wait for first reading
            time.sleep(0.5)
            if self.latest_reading:
                self.initialized = True
                logger.info(f"Real IMU sensor initialized at {self.port}")
            else:
                logger.warning("IMU initialized but no data received yet")
                
        except Exception as e:
            logger.warning(f"Could not initialize IMU: {e}, using simulation mode")
            
    def _read_loop(self):
        """Background thread to continuously read IMU data."""
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    # Try to parse as JSON (common format)
                    try:
                        data = json.loads(line)
                        self.latest_reading = data
                        self.data_queue.put(data)
                        # Keep queue size limited
                        while self.data_queue.qsize() > 100:
                            self.data_queue.get_nowait()
                    except json.JSONDecodeError:
                        # Try CSV format: ax,ay,az,gx,gy,gz
                        parts = line.split(',')
                        if len(parts) >= 6:
                            self.latest_reading = {
                                'ax': float(parts[0]),
                                'ay': float(parts[1]),
                                'az': float(parts[2]),
                                'gx': float(parts[3]),
                                'gy': float(parts[4]),
                                'gz': float(parts[5])
                            }
                            self.data_queue.put(self.latest_reading)
                            
            except Exception as e:
                logger.debug(f"IMU read error: {e}")
            time.sleep(0.01)
            
    def read(self, *, tick: int) -> float:
        """
        Read IMU data and return activity/motion confidence [0,1].
        Higher values indicate more motion/activity.
        """
        if not self.initialized or not self.latest_reading:
            # Fallback to simulated value
            import math
            # Simulate motion bursts
            burst = 0.8 if (tick % 40 in range(10, 15)) else 0.1
            return burst + 0.1 * abs(math.cos(tick / 15.0))
            
        try:
            # Get latest accelerometer data
            ax = self.latest_reading.get('ax', 0.0)
            ay = self.latest_reading.get('ay', 0.0)
            az = self.latest_reading.get('az', 0.0)
            
            # Get gyroscope data if available
            gx = self.latest_reading.get('gx', 0.0)
            gy = self.latest_reading.get('gy', 0.0)
            gz = self.latest_reading.get('gz', 0.0)
            
            # Update gravity estimate (complementary filter)
            accel = np.array([ax, ay, az])
            self.gravity = self.gravity_filter * self.gravity + (1 - self.gravity_filter) * accel
            
            # Remove gravity to get linear acceleration
            linear_accel = accel - self.gravity
            
            # Calculate motion magnitude
            accel_mag = np.linalg.norm(linear_accel)
            gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            
            # Combine accelerometer and gyroscope signals
            # Normalize to roughly [0,1] range
            motion_score = min(1.0, (accel_mag * self.motion_scale + 
                                    gyro_mag * 0.01))  # Gyro typically in deg/s
            
            # Add a baseline for sensor being active
            confidence = 0.1 + 0.9 * motion_score
            
            return confidence
            
        except Exception as e:
            logger.error(f"IMU sensor read error: {e}")
            return 0.0
            
    def get_raw_data(self) -> Dict[str, Any]:
        """Get raw IMU data for debugging."""
        return self.latest_reading.copy()
        
    def __del__(self):
        """Clean up serial connection."""
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)
        if self.serial_conn:
            self.serial_conn.close()