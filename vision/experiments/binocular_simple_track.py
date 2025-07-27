#!/usr/bin/env python3
"""
Simplified binocular tracking to debug focus movement
"""

import cv2
import numpy as np
import time
from collections import deque

class SimpleTrackingEye:
    """Simplified eye with basic motion tracking"""
    
    def __init__(self, eye_id, sensor_id):
        self.eye_id = eye_id
        self.sensor_id = sensor_id
        
        # Focus position (0-1 normalized)
        self.focus_x = 0.5
        self.focus_y = 0.5
        self.focus_radius = 0.15
        
        # Motion detection
        self.prev_gray = None
        self.motion_history = deque(maxlen=5)
        
        # Simple thresholds
        self.motion_threshold = 20  # Pixel difference threshold
        
    def process_frame(self, frame):
        """Process frame and update focus"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # Simple motion detection
            diff = cv2.absdiff(gray, self.prev_gray)
            
            # Find regions with motion
            _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours of motion
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest motion area
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                
                if M["m00"] > 100:  # Minimum area threshold
                    # Calculate center of motion
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Normalize to 0-1
                    new_x = cx / w
                    new_y = cy / h
                    
                    # Smooth movement
                    alpha = 0.3
                    self.focus_x = alpha * new_x + (1 - alpha) * self.focus_x
                    self.focus_y = alpha * new_y + (1 - alpha) * self.focus_y
                    
                    # Debug print
                    print(f"{self.eye_id}: Motion at ({cx}, {cy}) -> Focus ({self.focus_x:.2f}, {self.focus_y:.2f})")
        
        self.prev_gray = gray
        
        # Visualize
        result = frame.copy()
        
        # Draw focus circle
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        radius_px = int(self.focus_radius * min(h, w))
        
        color = (255, 100, 0) if self.eye_id == "left" else (0, 100, 255)
        cv2.circle(result, (focus_px, focus_py), radius_px, color, 3)
        
        # Draw crosshair
        cv2.line(result, (focus_px - 20, focus_py), (focus_px + 20, focus_py), color, 2)
        cv2.line(result, (focus_px, focus_py - 20), (focus_px, focus_py + 20), color, 2)
        
        # Labels
        cv2.putText(result, f"{self.eye_id.upper()} EYE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(result, f"Focus: ({self.focus_x:.2f}, {self.focus_y:.2f})", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show motion threshold areas
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            # Convert to color and blend
            thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            thresh_color[:,:,0] = 0  # Remove blue channel
            thresh_color[:,:,1] = 0  # Remove green channel
            result = cv2.addWeighted(result, 0.7, thresh_color, 0.3, 0)
        
        return result

def main():
    print("🎯 Simple Binocular Tracking Test")
    print("="*40)
    print("Testing basic motion tracking")
    print("Red overlay shows detected motion")
    print("Circles should follow motion")
    print("\nPress 'q' to quit")
    print("Press 'm' to create test motion")
    
    # Create simple eyes
    left_eye = SimpleTrackingEye("left", 0)
    right_eye = SimpleTrackingEye("right", 1)
    
    # Camera setup
    def get_pipeline(sensor_id):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1"
        )
    
    cap0 = cv2.VideoCapture(get_pipeline(0), cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(get_pipeline(1), cv2.CAP_GSTREAMER)
    
    if not cap0.isOpened() or not cap1.isOpened():
        print("Failed to open cameras")
        return
        
    # Test motion
    add_motion = False
    motion_pos = (320, 240)
    
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if ret0 and ret1:
            # Add test motion if requested
            if add_motion:
                cv2.circle(frame0, motion_pos, 50, (255, 255, 255), -1)
                cv2.circle(frame1, motion_pos, 50, (255, 255, 255), -1)
                add_motion = False
            
            # Process frames
            left_vis = left_eye.process_frame(frame0)
            right_vis = right_eye.process_frame(frame1)
            
            # Combine
            stereo = np.hstack([left_vis, right_vis])
            
            cv2.putText(stereo, "Simple Motion Tracking - Circles Should Follow Movement", 
                       (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Simple Tracking', stereo)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                add_motion = True
                # Random position for test motion
                motion_pos = (np.random.randint(100, 540), np.random.randint(100, 380))
                print(f"Adding test motion at {motion_pos}")
                
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()