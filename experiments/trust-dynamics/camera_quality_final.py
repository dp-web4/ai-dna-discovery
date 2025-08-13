#!/usr/bin/env python3
"""
Camera Quality Detection - Final Implementation
Based on CAMERA_QUALITY_DESIGN.md
"""

import cv2
import numpy as np

def assess_camera_quality(frame):
    """
    Assess camera quality based on design document.
    Returns quality score 0-1 and debug metrics.
    """
    if frame is None:
        return 0.0, None
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Calculate raw metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus = laplacian.var()
    
    # Store metrics for debugging
    metrics = {
        "brightness": brightness,
        "contrast": contrast,
        "edge_ratio": edge_ratio,
        "focus": focus
    }
    
    # 2. Early exit for obvious occlusion
    if brightness < 10:
        return brightness / 10.0, metrics  # Max 0.1
    if contrast < 5:
        return contrast / 5.0, metrics  # Max 0.1
    
    # 3. Convert metrics to scores (0-1 range)
    
    # Brightness score (most important)
    if brightness < 20:
        brightness_score = 0.5 * (brightness / 20.0)  # 0-0.5
    elif brightness < 50:
        brightness_score = 0.5 + 0.5 * ((brightness - 20) / 30.0)  # 0.5-1.0
    elif brightness <= 150:
        brightness_score = 1.0
    else:
        # Slight penalty for overexposure
        brightness_score = max(0.7, 1.0 - (brightness - 150) / 200.0)
    
    # Contrast score
    if contrast < 10:
        contrast_score = 0.3 * (contrast / 10.0)  # 0-0.3
    elif contrast < 40:
        contrast_score = 0.3 + 0.7 * ((contrast - 10) / 30.0)  # 0.3-1.0
    else:
        contrast_score = 1.0
    
    # Edge score
    if edge_ratio < 0.01:
        edge_score = 0.0  # Almost no edges
    elif edge_ratio < 0.1:
        edge_score = (edge_ratio - 0.01) / 0.09  # Ramp up
    else:
        edge_score = 1.0  # Normal or high edges
    
    # Focus score (least reliable)
    if focus < 1000:
        focus_score = 0.0
    elif focus < 10000:
        focus_score = (focus - 1000) / 9000.0
    else:
        focus_score = 1.0
    
    # 4. Weighted average
    quality = (
        brightness_score * 0.4 +
        contrast_score * 0.3 +
        edge_score * 0.2 +
        focus_score * 0.1
    )
    
    # Add scores to metrics for debugging
    metrics["scores"] = {
        "brightness": brightness_score,
        "contrast": contrast_score,
        "edge": edge_score,
        "focus": focus_score
    }
    
    return quality, metrics


def test_camera_quality():
    """Interactive test of camera quality detection"""
    
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
    print("CAMERA QUALITY TEST - FINAL")
    print("="*60)
    print("\nExpected behavior:")
    print("- Normal scene: 0.8-1.0")
    print("- Covered camera: 0.0-0.2")
    print("- Partial cover: 0.3-0.7")
    print("\nPress 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        quality, metrics = assess_camera_quality(frame)
        
        # Print detailed info every 30 frames (1 second)
        if frame_count % 30 == 0:
            print(f"\n[Frame {frame_count}]")
            print(f"  Brightness: {metrics['brightness']:.1f} (score: {metrics['scores']['brightness']:.2f})")
            print(f"  Contrast:   {metrics['contrast']:.1f} (score: {metrics['scores']['contrast']:.2f})")
            print(f"  Edges:      {metrics['edge_ratio']:.3f} (score: {metrics['scores']['edge']:.2f})")
            print(f"  Focus:      {metrics['focus']:.0f} (score: {metrics['scores']['focus']:.2f})")
            print(f"  >>> QUALITY: {quality:.2f} <<<")
        
        # Visual display
        small = cv2.resize(frame, (640, 360))
        
        # Determine status
        if quality > 0.7:
            status = "NORMAL"
            color = (0, 255, 0)  # Green
        elif quality > 0.3:
            status = "DEGRADED"
            color = (0, 255, 255)  # Yellow
        else:
            status = "COVERED"
            color = (0, 0, 255)  # Red
        
        # Add text overlay
        cv2.putText(small, f"Quality: {quality:.2f}", (30, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(small, status, (30, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Show brightness as reference
        cv2.putText(small, f"Brightness: {metrics['brightness']:.0f}", (30, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Quality bar
        bar_width = int(580 * quality)
        cv2.rectangle(small, (30, 140), (30 + bar_width, 160), color, -1)
        cv2.rectangle(small, (30, 140), (610, 160), (255, 255, 255), 2)
        
        cv2.imshow("Camera Quality Test", small)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Test complete")


if __name__ == "__main__":
    test_camera_quality()