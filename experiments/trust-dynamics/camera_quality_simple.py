#!/usr/bin/env python3
"""
Simple Direct Camera Quality Detection
Focus: Actually detect when camera is covered
"""

import cv2
import numpy as np

def assess_camera_quality_simple(frame):
    """
    Simple approach: Look for what actually changes when covered
    """
    if frame is None:
        return 0.0, {}
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Key insight: When covered, the image becomes:
    # 1. Very uniform (low standard deviation)
    # 2. Often darker (but not always - depends on cover material)
    # 3. No high-frequency content (no edges/details)
    
    metrics = {}
    
    # 1. UNIFORMITY CHECK - Most reliable
    # Standard deviation across whole image
    std_dev = np.std(gray)
    metrics['std_dev'] = std_dev
    
    # When covered, std_dev drops dramatically (< 10)
    # Normal scenes have std_dev > 30
    if std_dev < 5:
        uniformity_score = 0.0  # Extremely uniform
    elif std_dev < 15:
        uniformity_score = (std_dev - 5) / 10  # Scale 0-1
    else:
        uniformity_score = 1.0
    
    # 2. HIGH-FREQUENCY CONTENT - Laplacian
    # This measures focus/sharpness
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_std = np.std(laplacian)
    metrics['lap_std'] = lap_std
    
    # Covered camera has very low Laplacian std (< 5)
    # Normal scenes > 20
    if lap_std < 3:
        sharpness_score = 0.0
    elif lap_std < 10:
        sharpness_score = (lap_std - 3) / 7
    else:
        sharpness_score = 1.0
    
    # 3. UNIQUE PIXEL VALUES - Information content
    # Count unique values in 8x8 blocks
    h, w = gray.shape
    block_size = 32
    unique_counts = []
    
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            unique = len(np.unique(block))
            unique_counts.append(unique)
    
    avg_unique = np.mean(unique_counts) if unique_counts else 0
    metrics['avg_unique'] = avg_unique
    
    # Covered has very few unique values per block (< 10)
    # Normal has many (> 50)
    if avg_unique < 10:
        diversity_score = 0.0
    elif avg_unique < 50:
        diversity_score = (avg_unique - 10) / 40
    else:
        diversity_score = 1.0
    
    # 4. MEAN ABSOLUTE DEVIATION - Another uniformity measure
    mad = np.mean(np.abs(gray - np.mean(gray)))
    metrics['mad'] = mad
    
    if mad < 5:
        mad_score = 0.0
    elif mad < 20:
        mad_score = (mad - 5) / 15
    else:
        mad_score = 1.0
    
    # Store scores
    metrics['scores'] = {
        'uniformity': uniformity_score,
        'sharpness': sharpness_score,
        'diversity': diversity_score,
        'mad': mad_score
    }
    
    # Final quality - if ANY indicator shows covered, drop quality
    quality = min(uniformity_score, sharpness_score, diversity_score, mad_score)
    
    return quality, metrics


def test_simple_quality():
    """Test simple quality detection"""
    
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
    print("SIMPLE CAMERA QUALITY TEST")
    print("="*60)
    print("\nFocusing on what actually changes when covered:")
    print("- Image becomes uniform (low std deviation)")
    print("- Loss of high-frequency content")
    print("- Fewer unique pixel values")
    print("\nPress 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        quality, metrics = assess_camera_quality_simple(frame)
        
        # Print every 10 frames
        if frame_count % 10 == 0:
            print(f"\n[Frame {frame_count}]")
            print(f"  Std Dev:     {metrics['std_dev']:.1f} → {metrics['scores']['uniformity']:.2f}")
            print(f"  Laplacian:   {metrics['lap_std']:.1f} → {metrics['scores']['sharpness']:.2f}")
            print(f"  Unique Vals: {metrics['avg_unique']:.1f} → {metrics['scores']['diversity']:.2f}")
            print(f"  MAD:         {metrics['mad']:.1f} → {metrics['scores']['mad']:.2f}")
            print(f"  >>> QUALITY: {quality:.2f} <<<")
            
            # Diagnosis
            if quality < 0.3:
                issues = []
                for name, score in metrics['scores'].items():
                    if score < 0.3:
                        issues.append(name)
                print(f"  DETECTED: {', '.join(issues)}")
        
        # Visual display
        small = cv2.resize(frame, (640, 360))
        
        # Status
        if quality > 0.7:
            status = "NORMAL"
            color = (0, 255, 0)
        elif quality > 0.3:
            status = "PARTIAL"
            color = (0, 255, 255)
        else:
            status = "COVERED"
            color = (0, 0, 255)
        
        cv2.putText(small, f"Quality: {quality:.2f}", (30, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(small, status, (30, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Show key metric
        cv2.putText(small, f"StdDev: {metrics['std_dev']:.1f}", (30, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(small, f"Laplacian: {metrics['lap_std']:.1f}", (30, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Individual score bars
        y = 180
        for name, score in metrics['scores'].items():
            cv2.putText(small, name[:8], (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            bar_width = int(150 * score)
            bar_color = (0, 200, 0) if score > 0.5 else (0, 0, 200)
            cv2.rectangle(small, (120, y-10), (120+bar_width, y), bar_color, -1)
            cv2.rectangle(small, (120, y-10), (270, y), (100, 100, 100), 1)
            y += 20
        
        cv2.imshow("Simple Quality Test", small)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Test complete")


if __name__ == "__main__":
    test_simple_quality()