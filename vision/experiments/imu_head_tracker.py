#!/usr/bin/env python3
"""
IMU-Based Head Tracking for Binocular Vision
Predicts gaze direction and stabilizes attention based on head movement
"""

import cv2
import numpy as np
import threading
import time
import serial
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import math
import sys

# Add IMU path
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../imu'))

@dataclass
class HeadPose:
    """Head pose estimation from IMU"""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0
    
    def predict_future_pose(self, dt: float) -> 'HeadPose':
        """Predict head pose after dt seconds"""
        # Simple linear prediction
        future_roll = self.roll + self.angular_velocity[0] * dt
        future_pitch = self.pitch + self.angular_velocity[1] * dt
        future_yaw = self.yaw + self.angular_velocity[2] * dt
        
        return HeadPose(
            roll=future_roll,
            pitch=future_pitch,
            yaw=future_yaw,
            angular_velocity=self.angular_velocity.copy(),
            linear_acceleration=self.linear_acceleration.copy(),
            timestamp=self.timestamp + dt
        )

class MotionPredictor:
    """Predicts head motion patterns"""
    
    def __init__(self, history_size=30):
        self.pose_history = deque(maxlen=history_size)
        self.velocity_filter = deque(maxlen=5)
        
    def update(self, pose: HeadPose):
        """Update motion history"""
        self.pose_history.append(pose)
        
        # Calculate smoothed angular velocity
        if len(self.pose_history) >= 2:
            dt = pose.timestamp - self.pose_history[-2].timestamp
            if dt > 0:
                droll = pose.roll - self.pose_history[-2].roll
                dpitch = pose.pitch - self.pose_history[-2].pitch
                dyaw = pose.yaw - self.pose_history[-2].yaw
                
                # Handle angle wrapping
                droll = (droll + 180) % 360 - 180
                dpitch = (dpitch + 180) % 360 - 180
                dyaw = (dyaw + 180) % 360 - 180
                
                velocity = np.array([droll/dt, dpitch/dt, dyaw/dt])
                self.velocity_filter.append(velocity)
                
    def get_smoothed_velocity(self) -> np.ndarray:
        """Get filtered angular velocity"""
        if len(self.velocity_filter) == 0:
            return np.zeros(3)
        return np.mean(self.velocity_filter, axis=0)
        
    def detect_saccade(self) -> bool:
        """Detect rapid head movement (saccade-like)"""
        if len(self.velocity_filter) < 2:
            return False
            
        current_speed = np.linalg.norm(self.get_smoothed_velocity())
        # Threshold for saccade detection (degrees/second)
        return current_speed > 50.0
        
    def predict_stopping_point(self) -> Optional[Tuple[float, float, float]]:
        """Predict where head will stop after current motion"""
        if len(self.pose_history) < 5:
            return None
            
        velocity = self.get_smoothed_velocity()
        speed = np.linalg.norm(velocity)
        
        if speed < 5.0:  # Already slow/stopped
            return None
            
        # Simple deceleration model
        decel_time = speed / 100.0  # Assume 100 deg/s² deceleration
        
        current_pose = self.pose_history[-1]
        predicted = current_pose.predict_future_pose(decel_time * 0.5)
        
        return (predicted.roll, predicted.pitch, predicted.yaw)

class GazeEstimator:
    """Estimates gaze direction from head pose"""
    
    def __init__(self, fov_horizontal=60, fov_vertical=45):
        self.fov_h = fov_horizontal
        self.fov_v = fov_vertical
        self.gaze_history = deque(maxlen=10)
        
    def head_to_gaze(self, head_pose: HeadPose) -> Tuple[float, float]:
        """Convert head pose to gaze direction in image space"""
        # Simplified model: gaze follows head with slight lag
        # In reality, eyes can move independently
        
        # Map head angles to normalized gaze coordinates
        gaze_x = np.clip(head_pose.yaw / (self.fov_h / 2), -1, 1)
        gaze_y = np.clip(-head_pose.pitch / (self.fov_v / 2), -1, 1)
        
        self.gaze_history.append((gaze_x, gaze_y))
        
        # Smooth gaze
        if len(self.gaze_history) > 0:
            smooth_gaze = np.mean(self.gaze_history, axis=0)
            return float(smooth_gaze[0]), float(smooth_gaze[1])
        
        return gaze_x, gaze_y
        
    def gaze_to_image_coords(self, gaze_x: float, gaze_y: float, 
                           img_width: int, img_height: int) -> Tuple[int, int]:
        """Convert normalized gaze to image coordinates"""
        x = int((gaze_x + 1) * img_width / 2)
        y = int((gaze_y + 1) * img_height / 2)
        return np.clip(x, 0, img_width-1), np.clip(y, 0, img_height-1)

class AttentionController:
    """Controls visual attention based on head tracking"""
    
    def __init__(self, img_width=640, img_height=480):
        self.img_width = img_width
        self.img_height = img_height
        self.attention_point = (img_width // 2, img_height // 2)
        self.attention_radius = 100
        self.saccade_target = None
        self.attention_map = np.zeros((img_height, img_width), dtype=np.float32)
        
    def update_attention(self, gaze_point: Tuple[int, int], is_saccading: bool):
        """Update attention based on gaze and motion state"""
        if is_saccading and self.saccade_target is None:
            # Start saccade - lock current attention
            self.saccade_target = self.attention_point
        elif not is_saccading:
            # Follow gaze smoothly
            self.saccade_target = None
            alpha = 0.3  # Smoothing factor
            self.attention_point = (
                int(alpha * gaze_point[0] + (1-alpha) * self.attention_point[0]),
                int(alpha * gaze_point[1] + (1-alpha) * self.attention_point[1])
            )
            
        # Update attention map
        self.attention_map *= 0.9  # Decay
        y, x = np.ogrid[:self.img_height, :self.img_width]
        dist = np.sqrt((x - self.attention_point[0])**2 + 
                      (y - self.attention_point[1])**2)
        gaussian = np.exp(-(dist**2) / (2 * self.attention_radius**2))
        self.attention_map = np.maximum(self.attention_map, gaussian)
        
    def apply_attention(self, frame: np.ndarray) -> np.ndarray:
        """Apply attention-based visual processing"""
        # Create attention overlay
        attention_color = cv2.applyColorMap(
            (self.attention_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Blend with original
        result = cv2.addWeighted(frame, 0.7, attention_color, 0.3, 0)
        
        # Draw attention center
        cv2.circle(result, self.attention_point, 5, (0, 255, 0), -1)
        cv2.circle(result, self.attention_point, self.attention_radius, 
                  (0, 255, 0), 2)
        
        return result

class IMUHeadTracker:
    """Main head tracking system"""
    
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        self.port = port
        self.baud = baud
        self.serial = None
        self.running = False
        self.thread = None
        
        # Components
        self.current_pose = HeadPose()
        self.motion_predictor = MotionPredictor()
        self.gaze_estimator = GazeEstimator()
        self.lock = threading.Lock()
        
    def start(self):
        """Start head tracking"""
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=0.1)
            self.running = True
            self.thread = threading.Thread(target=self._tracking_loop)
            self.thread.daemon = True
            self.thread.start()
            print(f"Head tracker started on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to start head tracker: {e}")
            return False
            
    def stop(self):
        """Stop tracking"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.serial:
            self.serial.close()
            
    def _tracking_loop(self):
        """Main tracking loop"""
        buffer = bytearray()
        
        while self.running:
            try:
                if self.serial.in_waiting:
                    buffer.extend(self.serial.read(self.serial.in_waiting))
                    
                # Parse IMU packets
                while len(buffer) >= 11:
                    if buffer[0] != 0x55:
                        buffer.pop(0)
                        continue
                        
                    packet = buffer[:11]
                    buffer = buffer[11:]
                    
                    # Update pose based on packet type
                    with self.lock:
                        if packet[1] == 0x53:  # Angle
                            self.current_pose.roll = struct.unpack('<h', packet[2:4])[0] / 32768.0 * 180.0
                            self.current_pose.pitch = struct.unpack('<h', packet[4:6])[0] / 32768.0 * 180.0
                            self.current_pose.yaw = struct.unpack('<h', packet[6:8])[0] / 32768.0 * 180.0
                            self.current_pose.timestamp = time.time()
                            
                        elif packet[1] == 0x52:  # Gyroscope
                            gx = struct.unpack('<h', packet[2:4])[0] / 32768.0 * 2000.0
                            gy = struct.unpack('<h', packet[4:6])[0] / 32768.0 * 2000.0
                            gz = struct.unpack('<h', packet[6:8])[0] / 32768.0 * 2000.0
                            self.current_pose.angular_velocity = np.array([gx, gy, gz])
                            
                        elif packet[1] == 0x51:  # Acceleration
                            ax = struct.unpack('<h', packet[2:4])[0] / 32768.0 * 16.0
                            ay = struct.unpack('<h', packet[4:6])[0] / 32768.0 * 16.0
                            az = struct.unpack('<h', packet[6:8])[0] / 32768.0 * 16.0
                            self.current_pose.linear_acceleration = np.array([ax, ay, az])
                            
                        # Update motion predictor
                        self.motion_predictor.update(self.current_pose)
                        
            except Exception as e:
                print(f"Tracking error: {e}")
                time.sleep(0.1)
                
    def get_head_state(self) -> Tuple[HeadPose, bool, Optional[Tuple[float, float]]]:
        """Get current head state
        Returns: (pose, is_saccading, gaze_point)
        """
        with self.lock:
            pose = HeadPose(
                roll=self.current_pose.roll,
                pitch=self.current_pose.pitch,
                yaw=self.current_pose.yaw,
                angular_velocity=self.current_pose.angular_velocity.copy(),
                linear_acceleration=self.current_pose.linear_acceleration.copy(),
                timestamp=self.current_pose.timestamp
            )
            
            is_saccading = self.motion_predictor.detect_saccade()
            gaze = self.gaze_estimator.head_to_gaze(pose)
            
            return pose, is_saccading, gaze

def demo_head_tracking():
    """Demonstrate head tracking with visualization"""
    print("Head Tracking Demo")
    print("=" * 50)
    
    # Initialize components
    tracker = IMUHeadTracker()
    if not tracker.start():
        print("Failed to start tracker")
        return
        
    # Wait for initial data
    time.sleep(0.5)
    
    # Create demo window
    width, height = 800, 600
    attention_controller = AttentionController(width, height)
    
    cv2.namedWindow('Head Tracking Demo', cv2.WINDOW_NORMAL)
    
    print("\nTracking head movement...")
    print("Move your head to control attention")
    print("Press 'q' to quit")
    
    while True:
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get head state
        pose, is_saccading, gaze_norm = tracker.get_head_state()
        
        if gaze_norm:
            # Convert gaze to image coordinates
            gaze_point = tracker.gaze_estimator.gaze_to_image_coords(
                gaze_norm[0], gaze_norm[1], width, height
            )
            
            # Update attention
            attention_controller.update_attention(gaze_point, is_saccading)
            
            # Apply attention visualization
            frame = attention_controller.apply_attention(frame)
            
            # Draw head pose info
            info_y = 30
            cv2.putText(frame, f"Head: R:{pose.roll:.1f} P:{pose.pitch:.1f} Y:{pose.yaw:.1f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            info_y += 25
            velocity = tracker.motion_predictor.get_smoothed_velocity()
            speed = np.linalg.norm(velocity)
            cv2.putText(frame, f"Angular velocity: {speed:.1f} deg/s", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            info_y += 25
            state = "SACCADING" if is_saccading else "TRACKING"
            color = (0, 0, 255) if is_saccading else (0, 255, 0)
            cv2.putText(frame, f"State: {state}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw gaze point
            cv2.circle(frame, gaze_point, 10, (255, 255, 0), 2)
            cv2.line(frame, (width//2, height//2), gaze_point, (255, 255, 0), 1)
            
        # Display
        cv2.imshow('Head Tracking Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    tracker.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_head_tracking()