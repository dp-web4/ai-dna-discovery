#!/usr/bin/env python3
"""
Robust Camera Quality Detection
Uses multiple complementary techniques to detect occlusion
"""

import cv2
import numpy as np

def assess_camera_quality_robust(frame):
    """
    Multi-method approach to detect if camera is providing useful data
    """
    if frame is None:
        return 0.0, {}
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    metrics = {}
    scores = []
    
    # 1. HISTOGRAM ANALYSIS - Check if image has good distribution
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    
    # Entropy - higher = more information
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    metrics['entropy'] = entropy
    
    # Most pixels in narrow range = likely covered
    dominant_bin = np.max(hist)
    metrics['dominant_bin'] = dominant_bin
    
    # Score based on entropy (typical: 5-7 for normal, <3 for covered)
    if entropy < 2:
        entropy_score = 0.0
    elif entropy < 4:
        entropy_score = (entropy - 2) / 2
    else:
        entropy_score = 1.0
    scores.append(('entropy', entropy_score, 0.3))
    
    # 2. TEXTURE ANALYSIS - Look for actual patterns
    # Divide image into blocks and check variance
    block_size = 64
    block_variances = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            block_variances.append(np.var(block))
    
    # How many blocks have significant variance?
    significant_blocks = np.sum(np.array(block_variances) > 100)
    total_blocks = len(block_variances)
    texture_ratio = significant_blocks / total_blocks if total_blocks > 0 else 0
    metrics['texture_ratio'] = texture_ratio
    
    # Score (need at least 20% blocks with texture)
    texture_score = min(texture_ratio * 5, 1.0)
    scores.append(('texture', texture_score, 0.3))
    
    # 3. COLOR CHANNEL ANALYSIS - Check if all channels similar (gray/black)
    if len(frame.shape) == 3:
        b, g, r = cv2.split(frame)
        # Check correlation between channels
        bg_corr = np.corrcoef(b.flatten(), g.flatten())[0, 1]
        gr_corr = np.corrcoef(g.flatten(), r.flatten())[0, 1]
        rb_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
        avg_corr = (bg_corr + gr_corr + rb_corr) / 3
        metrics['color_correlation'] = avg_corr
        
        # High correlation (>0.95) suggests grayscale/covered
        if avg_corr > 0.98:
            color_score = 0.0  # Extremely correlated = likely covered
        elif avg_corr > 0.95:
            color_score = (0.98 - avg_corr) / 0.03
        else:
            color_score = 1.0
        scores.append(('color', color_score, 0.2))
    
    # 4. GRADIENT MAGNITUDE - Look for edges at multiple scales
    # Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # What percentage of pixels have significant gradients?
    significant_gradients = np.sum(magnitude > 20) / magnitude.size
    metrics['gradient_ratio'] = significant_gradients
    
    # Score (need at least 5% pixels with edges)
    if significant_gradients < 0.01:
        gradient_score = 0.0
    elif significant_gradients < 0.05:
        gradient_score = significant_gradients * 20
    else:
        gradient_score = 1.0
    scores.append(('gradient', gradient_score, 0.2))
    
    # 5. ABSOLUTE DARKNESS CHECK - Simple but effective
    mean_brightness = np.mean(gray)
    metrics['brightness'] = mean_brightness
    
    if mean_brightness < 5:
        brightness_score = 0.0  # Essentially black
    elif mean_brightness < 20:
        brightness_score = mean_brightness / 20
    else:
        brightness_score = 1.0  # Don't penalize bright scenes
    
    # Override everything if it's just dark
    if mean_brightness < 10:
        return brightness_score * 0.1, metrics
    
    # Weighted combination
    total_score = 0
    for name, score, weight in scores:
        metrics[f'{name}_score'] = score
        total_score += score * weight
    
    return total_score, metrics


def test_robust_quality():
    """Test the robust quality detection"""
    
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
    print("ROBUST CAMERA QUALITY TEST")
    print("="*60)
    print("\nThis should detect:")
    print("- Covered camera (dark/uniform)")
    print("- Low information content")
    print("- Lack of texture/edges")
    print("\nPress 'q' to quit\n")
    
    frame_count = 0
    quality_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        quality, metrics = assess_camera_quality_robust(frame)
        quality_history.append(quality)
        if len(quality_history) > 30:
            quality_history.pop(0)
        
        # Print every 15 frames
        if frame_count % 15 == 0:
            print(f"\n[Frame {frame_count}]")
            print(f"  Entropy:  {metrics.get('entropy', 0):.2f} (score: {metrics.get('entropy_score', 0):.2f})")
            print(f"  Texture:  {metrics.get('texture_ratio', 0):.2f} (score: {metrics.get('texture_score', 0):.2f})")
            print(f"  Gradient: {metrics.get('gradient_ratio', 0):.3f} (score: {metrics.get('gradient_score', 0):.2f})")
            if 'color_correlation' in metrics:
                print(f"  Color:    {metrics.get('color_correlation', 0):.3f} (score: {metrics.get('color_score', 0):.2f})")
            print(f"  Bright:   {metrics.get('brightness', 0):.0f}")
            print(f"  >>> QUALITY: {quality:.2f} <<<")
        
        # Visual display
        small = cv2.resize(frame, (640, 360))
        
        # Determine status
        if quality > 0.6:
            status = "GOOD"
            color = (0, 255, 0)
        elif quality > 0.3:
            status = "DEGRADED"
            color = (0, 255, 255)
        else:
            status = "OCCLUDED"
            color = (0, 0, 255)
        
        # Main status
        cv2.putText(small, f"Quality: {quality:.2f}", (30, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(small, status, (30, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Quality history graph
        graph_y = 340
        graph_h = 100
        if len(quality_history) > 1:
            for i in range(1, len(quality_history)):
                x1 = 30 + (i-1) * 20
                x2 = 30 + i * 20
                y1 = graph_y - int(quality_history[i-1] * graph_h)
                y2 = graph_y - int(quality_history[i] * graph_h)
                cv2.line(small, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Graph baseline
        cv2.line(small, (30, graph_y), (630, graph_y), (100, 100, 100), 1)
        cv2.line(small, (30, graph_y - graph_h//2), (630, graph_y - graph_h//2), (50, 50, 50), 1)
        
        # Metrics bars
        y = 120
        for name, value in [
            ("Entropy", metrics.get('entropy_score', 0)),
            ("Texture", metrics.get('texture_score', 0)),
            ("Gradient", metrics.get('gradient_score', 0))
        ]:
            cv2.putText(small, name, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            bar_width = int(150 * value)
            cv2.rectangle(small, (120, y-10), (120+bar_width, y), (0, 200, 0), -1)
            cv2.rectangle(small, (120, y-10), (270, y), (100, 100, 100), 1)
            y += 25
        
        cv2.imshow("Robust Quality Test", small)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Test complete")


if __name__ == "__main__":
    test_robust_quality()