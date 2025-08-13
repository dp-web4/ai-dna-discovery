#!/usr/bin/env python3
"""
Test GPT's camera trust implementation with Jetson CSI cameras
"""

import cv2
import sys
import time
import numpy as np

# Import GPT's camera trust
sys.path.insert(0, '/home/sprout/ai-workspace/private-context/ai-dna-discovery/coherence-engine')
from camera_trust import camera_trust_score

def gst_pipeline(sensor_id=0):
    """GStreamer pipeline for CSI camera"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
        f"video/x-raw(memory:NVMM), width=1920, height=1080, "
        f"format=NV12, framerate=30/1 ! "
        f"nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

# Initialize camera
cap = cv2.VideoCapture(gst_pipeline(0), cv2.CAP_GSTREAMER)

print("\n" + "="*60)
print("GPT CAMERA TRUST TEST")
print("="*60)
print("\nTesting GPT's camera trust scoring...")
print("1. Normal scene should show 0.7-1.0")
print("2. Cover camera - should drop")
print("3. Uncover - should recover")
print("\nPress 'q' to quit\n")

prev_gray = None
frame_count = 0
trust_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Get trust score
    score, metrics = camera_trust_score(frame, prev_gray=prev_gray, resize_w=320)
    
    # Update prev_gray for temporal analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None and prev_gray.shape == gray.shape:
        prev_gray = gray
    else:
        prev_gray = cv2.resize(gray, (320, int(gray.shape[0] * 320 / gray.shape[1])))
    
    trust_history.append(score)
    if len(trust_history) > 30:
        trust_history.pop(0)
    
    # Print detailed metrics every 15 frames
    if frame_count % 15 == 0:
        print(f"\n[Frame {frame_count}]")
        print(f"  TRUST SCORE: {score:.3f}")
        print(f"  Sharpness:   {metrics.get('tenengrad', 0):.1f} / {metrics.get('lap_var', 0):.1f}")
        print(f"  Edge density: {metrics.get('edge_density', 0):.3f}")
        print(f"  Contrast:     {metrics.get('rms_contrast', 0):.3f}")
        print(f"  Saturation:   {metrics.get('sat_mean', 0):.3f}")
        print(f"  Clipping:     L={metrics.get('low_clip', 0):.3f} H={metrics.get('high_clip', 0):.3f}")
        print(f"  Noise:        {metrics.get('spatial_noise', 0):.3f}")
        
        # Diagnosis
        if score < 0.5:
            print("  >> LOW TRUST - Check camera!")
    
    # Visual display
    small = cv2.resize(frame, (640, 360))
    
    # Color based on trust
    if score > 0.7:
        status = "TRUSTED"
        color = (0, 255, 0)
    elif score > 0.5:
        status = "DEGRADED"
        color = (0, 255, 255)
    else:
        status = "UNTRUSTED"
        color = (0, 0, 255)
    
    # Main display
    cv2.putText(small, f"Trust: {score:.3f}", (30, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(small, status, (30, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Key metrics
    cv2.putText(small, f"Sharp: {metrics.get('tenengrad', 0):.0f}", (30, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(small, f"Edges: {metrics.get('edge_density', 0):.3f}", (30, 145),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(small, f"Contrast: {metrics.get('rms_contrast', 0):.3f}", (30, 170),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Trust history graph
    if len(trust_history) > 1:
        graph_y = 340
        graph_h = 100
        for i in range(1, len(trust_history)):
            x1 = 30 + (i-1) * 20
            x2 = 30 + i * 20
            y1 = graph_y - int(trust_history[i-1] * graph_h)
            y2 = graph_y - int(trust_history[i] * graph_h)
            cv2.line(small, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Graph baseline
    cv2.line(small, (30, 340), (630, 340), (100, 100, 100), 1)
    cv2.line(small, (30, 290), (630, 290), (50, 50, 50), 1)  # 0.5 line
    
    cv2.imshow("GPT Camera Trust Test", small)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

# Analysis
if trust_history:
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(f"Average trust: {np.mean(trust_history):.3f}")
    print(f"Min trust:     {np.min(trust_history):.3f}")
    print(f"Max trust:     {np.max(trust_history):.3f}")
    print(f"Std dev:       {np.std(trust_history):.3f}")

cap.release()
cv2.destroyAllWindows()
print("\n✓ Test complete")