#!/usr/bin/env python3
"""Quick fix for the calculation issues"""

import cv2
import numpy as np

def assess_camera_quality_fixed(frame):
    """Fixed quality assessment"""
    if frame is None:
        return 0.0
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Canny edges
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_ratio = edge_pixels / total_pixels
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    print(f"Edge ratio: {edge_ratio:.3f}, Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
    
    # FIXED: Normal scenes have 0.15-0.35 edge ratio
    # Covered camera has < 0.01
    edge_score = 1.0
    if edge_ratio < 0.01:  # Very few edges - likely covered
        edge_score = edge_ratio * 100  # Scale up
    elif edge_ratio < 0.10:  # Few edges
        edge_score = 0.5 + (edge_ratio - 0.01) * 5.5
    elif edge_ratio > 0.40:  # Too many edges
        edge_score = max(0.3, 1.0 - (edge_ratio - 0.40) * 2)
    
    # Brightness: darkness is the main indicator of occlusion
    brightness_score = 1.0
    if brightness < 20:  # Very dark - likely covered
        brightness_score = brightness / 20.0
    elif brightness > 245:  # Too bright
        brightness_score = (255 - brightness) / 10.0
        
    # Contrast: low contrast = covered
    contrast_score = min(contrast / 30.0, 1.0)
    
    quality = edge_score * 0.4 + brightness_score * 0.4 + contrast_score * 0.2
    
    print(f"Scores - Edge: {edge_score:.2f}, Bright: {brightness_score:.2f}, Contrast: {contrast_score:.2f}")
    print(f"Final quality: {quality:.2f}\n")
    
    return max(0.0, min(1.0, quality))

# Test
pipeline = (
    "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, "
    "format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

print("\nTesting fixed quality assessment...")
print("Try covering the camera to see quality drop\n")
print("Press 'q' to quit\n")

for i in range(300):
    ret, frame = cap.read()
    if ret and i % 15 == 0:  # Every 0.5 seconds
        print(f"--- Frame {i} ---")
        quality = assess_camera_quality_fixed(frame)
        
        # Show frame with quality
        small = cv2.resize(frame, (640, 360))
        cv2.putText(small, f"Quality: {quality:.2f}", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if quality > 0.5 else (0, 0, 255), 2)
        cv2.imshow("Quality Test Fixed", small)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()