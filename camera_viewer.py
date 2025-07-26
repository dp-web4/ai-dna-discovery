#!/usr/bin/env python3
"""
Simple camera viewer for Jetson Orin Nano with IMX219
Displays live camera feed in a window
Press 'q' or ESC to quit
"""

import cv2
import sys

def gstreamer_pipeline(
    camera_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    """Create GStreamer pipeline for CSI camera"""
    return (
        f"nvarguscamerasrc sensor-id={camera_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

def main():
    print("🎥 Jetson Camera Viewer")
    print("Press 'q' or ESC to quit")
    print("Press 's' to save a snapshot")
    
    # Create camera capture with GStreamer pipeline
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera opened successfully")
    
    snapshot_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Display FPS on frame
        cv2.putText(frame, "Press 'q' to quit, 's' for snapshot", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Jetson Camera Feed', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("Exiting...")
            break
        elif key == ord('s'):  # Save snapshot
            filename = f"snapshot_{snapshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved {filename}")
            snapshot_count += 1
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera viewer closed")

if __name__ == "__main__":
    main()