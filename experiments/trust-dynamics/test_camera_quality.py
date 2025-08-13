#!/usr/bin/env python3
"""
Camera Quality Detection Test
Focus: Get the quality assessment working correctly
- Should be ~1.0 for normal scene
- Should drop to ~0.0 when covered
"""

import cv2
import numpy as np
import time

def assess_camera_quality_debug(frame):
    """Debug version with all metrics visible"""
    if frame is None:
        return 0.0, {}
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # All our metrics
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_ratio = edge_pixels / total_pixels
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Laplacian variance (focus measure)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus = laplacian.var()
    
    metrics = {
        "edge_ratio": edge_ratio,
        "brightness": brightness,
        "contrast": contrast,
        "focus": focus
    }
    
    # Quality calculation - SIMPLE AND DIRECT
    # Main indicator: Is the camera covered?
    quality = 1.0
    
    # Dark = covered (most reliable indicator)
    if brightness < 10:
        quality *= (brightness / 10.0)
    
    # Low contrast = covered
    if contrast < 10:
        quality *= (contrast / 10.0)
    
    # Very few edges = likely covered
    # BUT normal scenes have 0.2-0.35 edge ratio!
    if edge_ratio < 0.01:  # Almost no edges
        quality *= (edge_ratio / 0.01)
    
    # Low focus = blurry/covered
    if focus < 100:
        quality *= (focus / 100.0)
    
    return quality, metrics

# Test with camera
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
print("CAMERA QUALITY TEST")
print("="*60)
print("\nInstructions:")
print("1. Start with normal scene - quality should be ~1.0")
print("2. Cover camera with hand - quality should drop to ~0.0")
print("3. Uncover - quality should return to ~1.0")
print("\nPress 'q' to quit\n")

# Collect data for analysis
normal_metrics = []
covered_metrics = []
collecting_mode = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue
        
    quality, metrics = assess_camera_quality_debug(frame)
    
    # Display metrics
    print(f"\rEdge: {metrics['edge_ratio']:.3f} | "
          f"Bright: {metrics['brightness']:.1f} | "
          f"Contrast: {metrics['contrast']:.1f} | "
          f"Focus: {metrics['focus']:.1f} | "
          f"QUALITY: {quality:.3f}", end="")
    
    # Visual feedback
    small = cv2.resize(frame, (640, 360))
    
    # Color based on quality
    if quality > 0.7:
        color = (0, 255, 0)  # Green - good
        status = "NORMAL"
    elif quality > 0.3:
        color = (0, 255, 255)  # Yellow - degraded
        status = "DEGRADED"
    else:
        color = (0, 0, 255)  # Red - covered
        status = "COVERED"
    
    cv2.putText(small, f"Quality: {quality:.2f} - {status}", (30, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Show individual metric bars
    y = 60
    for name, value, max_val in [
        ("Edge", metrics['edge_ratio'], 0.4),
        ("Bright", metrics['brightness'], 255),
        ("Contrast", metrics['contrast'], 100),
        ("Focus", metrics['focus']/1000, 10)
    ]:
        cv2.putText(small, f"{name}:", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bar_width = int(200 * min(value/max_val, 1.0))
        cv2.rectangle(small, (100, y-10), (100+bar_width, y), (0, 255, 0), -1)
        cv2.rectangle(small, (100, y-10), (300, y), (255, 255, 255), 1)
        y += 25
    
    cv2.imshow("Quality Test", small)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        print("\n\nCollecting NORMAL scene data...")
        collecting_mode = "normal"
    elif key == ord('c'):
        print("\n\nCollecting COVERED scene data...")
        collecting_mode = "covered"
    elif key == ord('s'):
        collecting_mode = None
        print("\n\nStopped collecting")
    
    # Collect data
    if collecting_mode == "normal":
        normal_metrics.append(metrics)
    elif collecting_mode == "covered":
        covered_metrics.append(metrics)

# Analysis
print("\n\n" + "="*60)
print("ANALYSIS")
print("="*60)

if normal_metrics:
    print("\nNORMAL SCENE (average of {} samples):".format(len(normal_metrics)))
    for key in ["edge_ratio", "brightness", "contrast", "focus"]:
        values = [m[key] for m in normal_metrics]
        print(f"  {key:12s}: {np.mean(values):.3f} (±{np.std(values):.3f})")

if covered_metrics:
    print("\nCOVERED CAMERA (average of {} samples):".format(len(covered_metrics)))
    for key in ["edge_ratio", "brightness", "contrast", "focus"]:
        values = [m[key] for m in covered_metrics]
        print(f"  {key:12s}: {np.mean(values):.3f} (±{np.std(values):.3f})")

if normal_metrics and covered_metrics:
    print("\nRATIO (normal/covered):")
    for key in ["edge_ratio", "brightness", "contrast", "focus"]:
        normal_avg = np.mean([m[key] for m in normal_metrics])
        covered_avg = np.mean([m[key] for m in covered_metrics])
        if covered_avg > 0:
            ratio = normal_avg / covered_avg
            print(f"  {key:12s}: {ratio:.1f}x difference")

cap.release()
cv2.destroyAllWindows()
print("\n✓ Test complete")