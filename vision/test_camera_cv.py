#!/usr/bin/env python3
"""
Test camera with OpenCV using proper Jetson pipeline
"""

import cv2
import sys
sys.path.append('utils')
from camera_utils import open_camera

def main():
    print("Testing camera with OpenCV...")
    
    # Open camera
    cap = open_camera(1280, 720, 30)
    
    if cap is None:
        print("❌ Failed to open camera")
        return
        
    print("Camera opened! Press 'q' to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read frame")
            break
            
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Camera Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        
        # Print FPS every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal frames: {frame_count}")

if __name__ == "__main__":
    main()