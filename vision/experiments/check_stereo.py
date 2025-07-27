#!/usr/bin/env python3
"""
Quick stereo camera check - capture one frame from each camera
"""

import cv2
import numpy as np

def capture_frame(sensor_id):
    """Capture a single frame from camera"""
    pipeline = (
        f"nvarguscamerasrc sensor-id={sensor_id} num-buffers=1 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None

print("📸 Checking stereo cameras...")

# Capture from both cameras
frame0 = capture_frame(0)
frame1 = capture_frame(1)

if frame0 is not None and frame1 is not None:
    print("✅ Both cameras working!")
    print(f"   Camera 0: {frame0.shape}")
    print(f"   Camera 1: {frame1.shape}")
    
    # Save test images
    cv2.imwrite("camera0_test.jpg", frame0)
    cv2.imwrite("camera1_test.jpg", frame1)
    print("📁 Saved test images: camera0_test.jpg, camera1_test.jpg")
    
    # Create side-by-side comparison
    stereo = np.hstack([cv2.resize(frame0, (640, 360)), 
                       cv2.resize(frame1, (640, 360))])
    cv2.imwrite("stereo_test.jpg", stereo)
    print("📁 Saved stereo comparison: stereo_test.jpg")
    
else:
    print("❌ Camera error:")
    if frame0 is None:
        print("   Camera 0 failed")
    if frame1 is None:
        print("   Camera 1 failed")