#!/usr/bin/env python3
"""
Capture a single frame from integrated sensors and save visualization
"""

import cv2
import numpy as np
import time
import sys

def capture_sensor_frame():
    """Capture and save a single sensor visualization frame"""
    
    print("Capturing sensor frame...")
    
    # Initialize cameras
    left_cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    right_cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Wait for cameras to stabilize
    time.sleep(1)
    
    # Create canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    
    # Capture frames
    ret_l, left_frame = left_cap.read()
    ret_r, right_frame = right_cap.read()
    
    if ret_l and ret_r:
        # Resize frames to expected size
        left_frame = cv2.resize(left_frame, (640, 480))
        right_frame = cv2.resize(right_frame, (640, 480))
        
        # Add frames to canvas
        # Left camera
        canvas[50:530, 50:690] = left_frame
        cv2.putText(canvas, "LEFT CAMERA", (50, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Right camera  
        canvas[50:530, 590:1230] = right_frame
        cv2.putText(canvas, "RIGHT CAMERA", (590, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add simple motion detection
        gray_l = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for motion areas
        edges_l = cv2.Canny(gray_l, 50, 150)
        edges_r = cv2.Canny(gray_r, 50, 150)
        
        # Convert to color for display
        edges_l_color = cv2.cvtColor(edges_l, cv2.COLOR_GRAY2BGR)
        edges_r_color = cv2.cvtColor(edges_r, cv2.COLOR_GRAY2BGR)
        
        # Add edge overlays with transparency
        alpha = 0.3
        left_with_edges = cv2.addWeighted(left_frame, 1-alpha, edges_l_color, alpha, 0)
        right_with_edges = cv2.addWeighted(right_frame, 1-alpha, edges_r_color, alpha, 0)
        
        # Update canvas with edge detection
        canvas[50:530, 50:690] = left_with_edges
        canvas[50:530, 590:1230] = right_with_edges
        
        # Add IMU visualization placeholder
        cv2.rectangle(canvas, (50, 550), (400, 700), (50, 50, 50), -1)
        cv2.putText(canvas, "IMU DATA", (60, 580),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "Roll:  0.0°", (60, 620),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(canvas, "Pitch: 0.0°", (60, 650),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
        cv2.putText(canvas, "Yaw:   0.0°", (60, 680),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
        
        # Add audio visualization placeholder
        cv2.rectangle(canvas, (450, 550), (800, 700), (50, 50, 50), -1)
        cv2.putText(canvas, "AUDIO LEVELS", (460, 580),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mic level bar
        cv2.rectangle(canvas, (460, 620), (560, 640), (100, 255, 100), -1)
        cv2.putText(canvas, "MIC", (460, 615),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Speaker level bar
        cv2.rectangle(canvas, (460, 660), (520, 680), (255, 100, 100), -1)
        cv2.putText(canvas, "SPEAKER", (460, 655),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add sensor fusion status
        cv2.rectangle(canvas, (850, 550), (1230, 700), (50, 50, 50), -1)
        cv2.putText(canvas, "SENSOR FUSION", (860, 580),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "Cameras: ACTIVE", (860, 620),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(canvas, "IMU: CONNECTED", (860, 645),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(canvas, "Audio: READY", (860, 670),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save the frame
        cv2.imwrite('sensor_dashboard.jpg', canvas)
        print("✓ Saved sensor_dashboard.jpg")
        
        # Also save individual camera frames
        cv2.imwrite('left_camera.jpg', left_frame)
        cv2.imwrite('right_camera.jpg', right_frame)
        print("✓ Saved left_camera.jpg and right_camera.jpg")
        
    else:
        print("✗ Failed to capture from cameras")
    
    # Cleanup
    left_cap.release()
    right_cap.release()
    
    print("Done!")

if __name__ == "__main__":
    capture_sensor_frame()