#!/usr/bin/env python3
"""
IMU-Stabilized Binocular Vision
Integrates Yahboom CMP10A IMU data with binocular vision for stabilization
"""

import cv2
import numpy as np
import threading
import time
import serial
import struct
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import sys

# Add IMU path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../imu'))

# Import orientation mapper
try:
    from imu_orientation_mapper import IMUOrientationMapper, OrientationConfig
    ORIENTATION_MAPPING_AVAILABLE = True
except ImportError:
    ORIENTATION_MAPPING_AVAILABLE = False

@dataclass
class IMUData:
    """IMU sensor readings"""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    timestamp: float = 0.0

class IMUReader:
    """Thread-safe IMU data reader"""
    
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        self.port = port
        self.baud = baud
        self.serial = None
        self.latest_data = IMUData()
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
    def start(self):
        """Start IMU reading thread"""
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=0.1)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop)
            self.thread.daemon = True
            self.thread.start()
            print(f"IMU reader started on {self.port} @ {self.baud}")
            return True
        except Exception as e:
            print(f"Failed to start IMU: {e}")
            return False
            
    def stop(self):
        """Stop IMU reading"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.serial:
            self.serial.close()
            
    def _read_loop(self):
        """Main IMU reading loop"""
        buffer = bytearray()
        
        while self.running:
            try:
                if self.serial.in_waiting:
                    buffer.extend(self.serial.read(self.serial.in_waiting))
                    
                # Look for 0x55 header
                while len(buffer) >= 11:
                    if buffer[0] != 0x55:
                        buffer.pop(0)
                        continue
                        
                    packet = buffer[:11]
                    buffer = buffer[11:]
                    
                    # Parse based on packet type
                    if packet[1] == 0x53:  # Angle packet
                        roll = struct.unpack('<h', packet[2:4])[0] / 32768.0 * 180.0
                        pitch = struct.unpack('<h', packet[4:6])[0] / 32768.0 * 180.0
                        yaw = struct.unpack('<h', packet[6:8])[0] / 32768.0 * 180.0
                        
                        with self.lock:
                            self.latest_data.roll = roll
                            self.latest_data.pitch = pitch
                            self.latest_data.yaw = yaw
                            self.latest_data.timestamp = time.time()
                            
                    elif packet[1] == 0x51:  # Acceleration packet
                        ax = struct.unpack('<h', packet[2:4])[0] / 32768.0 * 16.0
                        ay = struct.unpack('<h', packet[4:6])[0] / 32768.0 * 16.0
                        az = struct.unpack('<h', packet[6:8])[0] / 32768.0 * 16.0
                        
                        with self.lock:
                            self.latest_data.accel_x = ax
                            self.latest_data.accel_y = ay
                            self.latest_data.accel_z = az
                            
                    elif packet[1] == 0x52:  # Gyroscope packet
                        gx = struct.unpack('<h', packet[2:4])[0] / 32768.0 * 2000.0
                        gy = struct.unpack('<h', packet[4:6])[0] / 32768.0 * 2000.0
                        gz = struct.unpack('<h', packet[6:8])[0] / 32768.0 * 2000.0
                        
                        with self.lock:
                            self.latest_data.gyro_x = gx
                            self.latest_data.gyro_y = gy
                            self.latest_data.gyro_z = gz
                            
            except Exception as e:
                print(f"IMU read error: {e}")
                time.sleep(0.1)
                
    def get_data(self) -> IMUData:
        """Get latest IMU data (thread-safe)"""
        with self.lock:
            return IMUData(
                roll=self.latest_data.roll,
                pitch=self.latest_data.pitch,
                yaw=self.latest_data.yaw,
                accel_x=self.latest_data.accel_x,
                accel_y=self.latest_data.accel_y,
                accel_z=self.latest_data.accel_z,
                gyro_x=self.latest_data.gyro_x,
                gyro_y=self.latest_data.gyro_y,
                gyro_z=self.latest_data.gyro_z,
                timestamp=self.latest_data.timestamp
            )

class StabilizedBinocularVision:
    """Binocular vision with IMU-based stabilization"""
    
    def __init__(self, camera_left=0, camera_right=1):
        self.cap_left = None
        self.cap_right = None
        self.camera_left = camera_left
        self.camera_right = camera_right
        
        # IMU integration
        self.imu_reader = IMUReader()
        self.imu_available = False
        
        # Orientation mapping
        self.orientation_mapper = None
        if ORIENTATION_MAPPING_AVAILABLE:
            try:
                config = OrientationConfig.load()
                self.orientation_mapper = IMUOrientationMapper(config)
                print("Loaded IMU orientation configuration")
            except:
                print("No IMU orientation config found, using default")
        
        # Stabilization parameters
        self.stabilization_enabled = True
        self.rotation_history = deque(maxlen=10)  # Smooth rotation
        self.reference_orientation = None
        
        # Display parameters
        self.window_width = 1280
        self.window_height = 480
        
    def initialize_cameras(self):
        """Initialize both cameras"""
        # GStreamer pipeline for CSI cameras
        def gst_pipeline(sensor_id):
            return (
                f"nvarguscamerasrc sensor-id={sensor_id} ! "
                "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
                "nvvidconv flip-method=0 ! "
                "video/x-raw, width=640, height=480, format=BGR ! "
                "appsink drop=1"
            )
        
        self.cap_left = cv2.VideoCapture(gst_pipeline(self.camera_left), cv2.CAP_GSTREAMER)
        self.cap_right = cv2.VideoCapture(gst_pipeline(self.camera_right), cv2.CAP_GSTREAMER)
        
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            print("Failed to open cameras")
            return False
            
        print("Cameras initialized successfully")
        return True
        
    def initialize_imu(self):
        """Initialize IMU reader"""
        self.imu_available = self.imu_reader.start()
        if self.imu_available:
            # Wait for initial data
            time.sleep(0.5)
            # Set reference orientation
            self.reference_orientation = self.imu_reader.get_data()
            print("IMU initialized, reference orientation set")
        else:
            print("IMU not available, running without stabilization")
            
    def get_rotation_matrix(self, roll, pitch, yaw):
        """Create rotation matrix from Euler angles"""
        # Convert to radians
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        return Rz @ Ry @ Rx
        
    def stabilize_frame(self, frame, imu_data: IMUData):
        """Apply stabilization to frame based on IMU data"""
        if not self.stabilization_enabled or not self.reference_orientation:
            return frame
            
        # Apply orientation mapping if available
        if self.orientation_mapper:
            current_angles = self.orientation_mapper.map_angles(
                imu_data.roll, imu_data.pitch, imu_data.yaw
            )
            ref_angles = self.orientation_mapper.map_angles(
                self.reference_orientation.roll,
                self.reference_orientation.pitch,
                self.reference_orientation.yaw
            )
            delta_roll = current_angles[0] - ref_angles[0]
            delta_pitch = current_angles[1] - ref_angles[1]
            delta_yaw = current_angles[2] - ref_angles[2]
        else:
            # No mapping, use raw values
            delta_roll = imu_data.roll - self.reference_orientation.roll
            delta_pitch = imu_data.pitch - self.reference_orientation.pitch
            delta_yaw = imu_data.yaw - self.reference_orientation.yaw
        
        # Smooth rotation changes
        self.rotation_history.append((delta_roll, delta_pitch, delta_yaw))
        if len(self.rotation_history) > 0:
            avg_rotation = np.mean(self.rotation_history, axis=0)
            delta_roll, delta_pitch, delta_yaw = avg_rotation
            
        # Apply rotation compensation
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        # For now, apply 2D rotation for yaw compensation
        # (Full 3D stabilization would require camera calibration)
        M = cv2.getRotationMatrix2D(center, -delta_yaw, 1.0)
        
        # Add translation for pitch/roll compensation
        # Approximate head movement effect
        tx = -delta_roll * 2  # Pixels per degree
        ty = delta_pitch * 2
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply transformation
        stabilized = cv2.warpAffine(frame, M, (w, h), 
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(64, 64, 64))
        
        return stabilized
        
    def draw_imu_info(self, frame, imu_data: IMUData):
        """Draw IMU information overlay"""
        if not self.imu_available:
            return
            
        # IMU status area
        cv2.rectangle(frame, (10, 10), (250, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 120), (0, 255, 0), 2)
        
        # Draw IMU data
        y = 30
        cv2.putText(frame, "IMU Data:", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y += 20
        cv2.putText(frame, f"Roll:  {imu_data.roll:6.1f}°", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        y += 15
        cv2.putText(frame, f"Pitch: {imu_data.pitch:6.1f}°", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        y += 15
        cv2.putText(frame, f"Yaw:   {imu_data.yaw:6.1f}°", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Stabilization status
        y += 20
        status = "ON" if self.stabilization_enabled else "OFF"
        color = (0, 255, 0) if self.stabilization_enabled else (0, 0, 255)
        cv2.putText(frame, f"Stabilization: {status}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
    def run(self):
        """Main vision loop"""
        if not self.initialize_cameras():
            return
            
        self.initialize_imu()
        
        cv2.namedWindow('Stabilized Binocular Vision', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stabilized Binocular Vision', self.window_width, self.window_height)
        
        print("\nControls:")
        print("- 's': Toggle stabilization")
        print("- 'r': Reset reference orientation")
        print("- 'q': Quit")
        print("- ESC: Quit")
        
        fps_timer = time.time()
        frame_count = 0
        
        while True:
            # Capture frames
            ret_left, frame_left = self.cap_left.read()
            ret_right, frame_right = self.cap_right.read()
            
            if not ret_left or not ret_right:
                print("Failed to capture frames")
                break
                
            # Get IMU data
            imu_data = None
            if self.imu_available:
                imu_data = self.imu_reader.get_data()
                
                # Apply stabilization
                if imu_data and self.stabilization_enabled:
                    frame_left = self.stabilize_frame(frame_left, imu_data)
                    frame_right = self.stabilize_frame(frame_right, imu_data)
            
            # Create side-by-side view
            combined = np.hstack([frame_left, frame_right])
            
            # Draw overlays
            if imu_data:
                self.draw_imu_info(combined, imu_data)
                
            # FPS calculation
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_timer)
                fps_timer = time.time()
                
                # Draw FPS
                cv2.putText(combined, f"FPS: {fps:.1f}", (self.window_width - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
            # Draw center lines for alignment
            h, w = combined.shape[:2]
            cv2.line(combined, (w//4, 0), (w//4, h), (0, 255, 0), 1)
            cv2.line(combined, (3*w//4, 0), (3*w//4, h), (0, 255, 0), 1)
            cv2.line(combined, (0, h//2), (w, h//2), (0, 255, 0), 1)
            
            # Display
            cv2.imshow('Stabilized Binocular Vision', combined)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('s'):
                self.stabilization_enabled = not self.stabilization_enabled
                print(f"Stabilization: {'ON' if self.stabilization_enabled else 'OFF'}")
            elif key == ord('r'):
                if self.imu_available:
                    self.reference_orientation = self.imu_reader.get_data()
                    print("Reference orientation reset")
                    
        # Cleanup
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()
        if self.imu_reader:
            self.imu_reader.stop()
        cv2.destroyAllWindows()

def main():
    """Run stabilized binocular vision"""
    print("Starting IMU-Stabilized Binocular Vision")
    print("=" * 50)
    
    vision = StabilizedBinocularVision()
    
    try:
        vision.run()
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        vision.cleanup()

if __name__ == "__main__":
    main()