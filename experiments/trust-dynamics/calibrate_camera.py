#!/usr/bin/env python3
"""
Camera Calibration Tool
Captures actual data for covered vs uncovered states
"""

import cv2
import numpy as np
import time

def get_metrics(frame):
    """Extract raw metrics from frame"""
    if frame is None:
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Raw metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus = laplacian.var()
    
    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_ratio": edge_ratio,
        "focus": focus
    }

# Setup camera
pipeline = (
    "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, "
    "format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

print("\n" + "="*60)
print("CAMERA CALIBRATION")
print("="*60)
print("\nInstructions:")
print("1. Press 'n' to sample NORMAL scene")
print("2. Press 'c' to sample COVERED camera")
print("3. Press 'p' to sample PARTIAL cover")
print("4. Press 'a' to show analysis")
print("5. Press 'q' to quit")
print("\nStart by pressing 'n' with camera uncovered...")
print("="*60 + "\n")

# Storage for samples
samples = {
    "normal": [],
    "covered": [],
    "partial": []
}

current_mode = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Get current metrics
    metrics = get_metrics(frame)
    
    # Sampling mode
    if current_mode:
        samples[current_mode].append(metrics)
        print(f"\r[{current_mode.upper()}] Samples: {len(samples[current_mode])} | "
              f"Bright: {metrics['brightness']:.0f} | "
              f"Contrast: {metrics['contrast']:.0f} | "
              f"Edges: {metrics['edge_ratio']:.3f} | "
              f"Focus: {metrics['focus']:.0f}", end="")
    
    # Display
    small = cv2.resize(frame, (640, 360))
    
    # Show current metrics
    cv2.putText(small, f"Brightness: {metrics['brightness']:.0f}", (30, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(small, f"Contrast: {metrics['contrast']:.0f}", (30, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    if current_mode:
        color = {"normal": (0, 255, 0), "covered": (0, 0, 255), "partial": (0, 255, 255)}[current_mode]
        cv2.putText(small, f"SAMPLING: {current_mode.upper()}", (30, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Calibration", small)
    
    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_mode = "normal"
        print(f"\n\n>>> SAMPLING NORMAL - Keep camera uncovered <<<")
    elif key == ord('c'):
        current_mode = "covered"
        print(f"\n\n>>> SAMPLING COVERED - Cover camera completely <<<")
    elif key == ord('p'):
        current_mode = "partial"
        print(f"\n\n>>> SAMPLING PARTIAL - Partially cover camera <<<")
    elif key == ord('s'):
        current_mode = None
        print(f"\n\n>>> STOPPED SAMPLING <<<")
    elif key == ord('a'):
        # Analysis
        print("\n\n" + "="*60)
        print("ANALYSIS")
        print("="*60)
        
        for mode in ["normal", "covered", "partial"]:
            if samples[mode]:
                print(f"\n{mode.upper()} ({len(samples[mode])} samples):")
                
                for metric in ["brightness", "contrast", "edge_ratio", "focus"]:
                    values = [s[metric] for s in samples[mode]]
                    avg = np.mean(values)
                    std = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    print(f"  {metric:12s}: avg={avg:8.2f} (std={std:.2f}, range={min_val:.2f}-{max_val:.2f})")
        
        # Calculate thresholds
        if samples["normal"] and samples["covered"]:
            print("\n" + "-"*60)
            print("SUGGESTED THRESHOLDS:")
            
            for metric in ["brightness", "contrast", "edge_ratio", "focus"]:
                normal_vals = [s[metric] for s in samples["normal"]]
                covered_vals = [s[metric] for s in samples["covered"]]
                
                normal_min = np.min(normal_vals)
                covered_max = np.max(covered_vals)
                
                threshold = (normal_min + covered_max) / 2
                
                print(f"  {metric:12s}: < {threshold:.2f} indicates occlusion")
                print(f"                  (normal>{normal_min:.2f}, covered<{covered_max:.2f})")

cap.release()
cv2.destroyAllWindows()

# Save calibration data
if samples["normal"] and samples["covered"]:
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    
    calibration = {
        "normal": {
            "brightness": np.mean([s["brightness"] for s in samples["normal"]]),
            "contrast": np.mean([s["contrast"] for s in samples["normal"]]),
            "edge_ratio": np.mean([s["edge_ratio"] for s in samples["normal"]]),
            "focus": np.mean([s["focus"] for s in samples["normal"]])
        },
        "covered": {
            "brightness": np.mean([s["brightness"] for s in samples["covered"]]),
            "contrast": np.mean([s["contrast"] for s in samples["covered"]]),
            "edge_ratio": np.mean([s["edge_ratio"] for s in samples["covered"]]),
            "focus": np.mean([s["focus"] for s in samples["covered"]])
        }
    }
    
    import json
    with open("camera_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    
    print("Saved calibration to camera_calibration.json")
    print("\nUse these values to create proper quality assessment!")