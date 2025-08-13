#!/usr/bin/env python3
"""
Camera Quality Detection - FIXED
- Weighted average instead of multiplication
- Proper focus thresholds
- Better edge interpretation
"""

import cv2
import numpy as np

def assess_camera_quality_proper(frame):
    """Proper quality assessment with weighted average"""
    if frame is None:
        return 0.0, {}
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Metrics
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
    
    # Individual scores (0-1 range each)
    
    # 1. BRIGHTNESS: Main occlusion indicator
    # Normal: 80-100, Covered: < 20
    if brightness < 20:
        brightness_score = brightness / 20.0  # 0-1 for dark
    elif brightness > 200:
        brightness_score = (255 - brightness) / 55.0  # Penalty for too bright
    else:
        brightness_score = 1.0  # Normal range
    
    # 2. CONTRAST: Low = covered
    # Normal: 60-80, Covered: < 20
    if contrast < 20:
        contrast_score = contrast / 20.0
    elif contrast < 60:
        contrast_score = 0.5 + (contrast - 20) / 80.0  # Ramp up
    else:
        contrast_score = 1.0
    
    # 3. EDGE RATIO: Too few OR too many = bad
    # Normal: 0.3-0.35, Covered: < 0.01
    if edge_ratio < 0.01:
        edge_score = 0.0  # Almost no edges = covered
    elif edge_ratio < 0.1:
        edge_score = edge_ratio * 10  # Ramp up
    elif edge_ratio < 0.25:
        edge_score = 1.0  # Bit low but ok
    elif edge_ratio < 0.4:
        edge_score = 1.0  # Normal range
    else:
        edge_score = max(0.5, 1.0 - (edge_ratio - 0.4) * 2)  # Too many edges
    
    # 4. FOCUS: Your data shows 20k-120k normal, covered would be much lower
    if focus < 1000:
        focus_score = 0.0  # Very blurry/covered
    elif focus < 10000:
        focus_score = focus / 10000.0  # Ramp up
    elif focus < 200000:
        focus_score = 1.0  # Normal range
    else:
        focus_score = 0.8  # Extremely sharp (unusual)
    
    # WEIGHTED AVERAGE (not multiplication!)
    quality = (
        brightness_score * 0.35 +  # Most reliable
        contrast_score * 0.25 +
        edge_score * 0.20 +
        focus_score * 0.20
    )
    
    # Debug info
    scores = {
        "bright": brightness_score,
        "contrast": contrast_score,
        "edge": edge_score,
        "focus": focus_score
    }
    
    return quality, metrics, scores

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

print("\n" + "="*60)
print("CAMERA QUALITY TEST - FIXED")
print("="*60)
print("\nInstructions:")
print("1. Normal scene should show ~1.0")
print("2. Cover camera - should drop to ~0.0")
print("3. Uncover - should return to ~1.0")
print("\nPress 'q' to quit\n")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    quality, metrics, scores = assess_camera_quality_proper(frame)
    
    # Print every 10 frames
    if frame_count % 10 == 0:
        print(f"\n--- Frame {frame_count} ---")
        print(f"Raw:   Edge={metrics['edge_ratio']:.3f}, Bright={metrics['brightness']:.0f}, "
              f"Contrast={metrics['contrast']:.0f}, Focus={metrics['focus']:.0f}")
        print(f"Score: Edge={scores['edge']:.2f}, Bright={scores['bright']:.2f}, "
              f"Contrast={scores['contrast']:.2f}, Focus={scores['focus']:.2f}")
        print(f"QUALITY: {quality:.2f}")
    
    # Visual
    small = cv2.resize(frame, (640, 360))
    
    # Status based on quality
    if quality > 0.7:
        color = (0, 255, 0)
        status = "NORMAL"
    elif quality > 0.3:
        color = (0, 255, 255)
        status = "DEGRADED"
    else:
        color = (0, 0, 255)
        status = "COVERED"
    
    cv2.putText(small, f"Quality: {quality:.2f} - {status}", (30, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Show individual scores as bars
    y = 70
    for name, score in [("Bright", scores['bright']), 
                        ("Contrast", scores['contrast']),
                        ("Edge", scores['edge']), 
                        ("Focus", scores['focus'])]:
        cv2.putText(small, f"{name}:", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bar_width = int(200 * score)
        bar_color = (0, 255, 0) if score > 0.5 else (0, 0, 255)
        cv2.rectangle(small, (120, y-10), (120+bar_width, y), bar_color, -1)
        cv2.rectangle(small, (120, y-10), (320, y), (255, 255, 255), 1)
        cv2.putText(small, f"{score:.2f}", (330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 25
    
    cv2.imshow("Quality Test Fixed", small)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("\n✓ Test complete")