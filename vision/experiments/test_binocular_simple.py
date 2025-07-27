#!/usr/bin/env python3
"""
Simple test to debug binocular vision circles
"""

import cv2
import numpy as np
import time

def process_eye(frame, eye_name, color):
    """Simple visualization with circle"""
    h, w = frame.shape[:2]
    
    # Always draw a circle in the center for testing
    center_x = w // 2
    center_y = h // 2
    radius = int(0.15 * min(h, w))
    
    # Draw focus circle
    cv2.circle(frame, (center_x, center_y), radius, color, 3)
    
    # Add label
    cv2.putText(frame, f"{eye_name} EYE", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Add timestamp to see it's updating
    cv2.putText(frame, f"Time: {time.time():.1f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    print("Simple binocular circle test")
    
    # Camera pipelines
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
        
    print("Cameras opened successfully")
    print("You should see orange circle on left, blue circle on right")
    print("Press 'q' to quit")
    
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if ret0 and ret1:
            # Process each eye
            left_vis = process_eye(frame0, "LEFT", (255, 100, 0))  # Orange
            right_vis = process_eye(frame1, "RIGHT", (0, 100, 255))  # Blue
            
            # Combine
            stereo = np.hstack([left_vis, right_vis])
            
            # Add title
            cv2.putText(stereo, "Binocular Vision Test - Circles Should Be Visible", 
                       (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Binocular Test', stereo)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()