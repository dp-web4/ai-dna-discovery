#!/usr/bin/env python3
"""
Minimal GPU version - focuses on reducing memory transfers
The key to GPU performance on Jetson!
"""

import cv2
import numpy as np
import time
from collections import deque

class MinimalGPUVision:
    def __init__(self):
        # Focus parameters
        self.focus_x = 0.5
        self.focus_y = 0.5
        self.focus_radius = 0.15
        
        # Motion detection
        self.motion_grid_size = 8
        self.ambient_motion = 0.016
        self.motion_heatmap = np.zeros((self.motion_grid_size, self.motion_grid_size))
        
        # Performance
        self.frame_times = deque(maxlen=30)
        self.prev_gray = None
        self.peaks = []
        
    def process_frame(self, frame):
        """Process with minimal overhead"""
        start = time.time()
        h, w = frame.shape[:2]
        
        # CPU operations - but optimized
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # Simple motion detection
            diff = cv2.absdiff(gray, self.prev_gray)
            
            # Update motion heatmap
            self._update_motion(diff, h, w)
            
            # Find peaks
            self._find_peaks()
            
            # Update focus
            if self.peaks and self.peaks[0]['pa_ratio'] > 2.0:
                self.focus_x = self.peaks[0]['x']
                self.focus_y = self.peaks[0]['y']
        
        self.prev_gray = gray
        
        # Visualize
        result = self._visualize(frame, h, w)
        
        self.frame_times.append(time.time() - start)
        
        return result
        
    def _update_motion(self, diff, h, w):
        """Fast motion grid update"""
        self.motion_heatmap *= 0.9
        
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        # Focus mask - simple distance check
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        focus_r2 = (self.focus_radius * min(h, w)) ** 2
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                # Cell center
                cx = (j + 0.5) * grid_w
                cy = (i + 0.5) * grid_h
                
                # Check if peripheral
                dx = cx - focus_px
                dy = cy - focus_py
                if dx*dx + dy*dy > focus_r2:
                    # Get cell motion
                    y1, y2 = i * grid_h, min((i + 1) * grid_h, h)
                    x1, x2 = j * grid_w, min((j + 1) * grid_w, w)
                    
                    cell_motion = np.mean(diff[y1:y2, x1:x2]) / 255.0
                    self.motion_heatmap[i, j] += cell_motion
    
    def _find_peaks(self):
        """Find motion peaks quickly"""
        self.peaks = []
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                if self.motion_heatmap[i, j] > self.ambient_motion * 1.5:
                    pa_ratio = self.motion_heatmap[i, j] / self.ambient_motion
                    self.peaks.append({
                        'x': (j + 0.5) / self.motion_grid_size,
                        'y': (i + 0.5) / self.motion_grid_size,
                        'pa_ratio': pa_ratio
                    })
        
        self.peaks.sort(key=lambda p: p['pa_ratio'], reverse=True)
    
    def _visualize(self, frame, h, w):
        """Minimal visualization"""
        result = frame.copy()
        
        # Focus circle
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        radius_px = int(self.focus_radius * min(h, w))
        cv2.circle(result, (focus_px, focus_py), radius_px, (0, 255, 255), 2)
        
        # Simple heatmap overlay
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                if self.motion_heatmap[i, j] > 0.01:
                    x1, y1 = j * grid_w, i * grid_h
                    x2, y2 = (j + 1) * grid_w, (i + 1) * grid_h
                    
                    intensity = min(255, int(self.motion_heatmap[i, j] * 5000))
                    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, intensity), 2)
        
        # Performance stats
        if len(self.frame_times) > 10:
            avg_time = np.mean(self.frame_times) * 1000
            fps = 1000 / avg_time
            cv2.putText(result, f"Optimized CPU: {avg_time:.1f}ms ({fps:.1f} FPS)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Peak indicators
        for peak in self.peaks[:3]:
            px = int(peak['x'] * w)
            py = int(peak['y'] * h)
            cv2.circle(result, (px, py), 10, (0, 0, 255), 2)
        
        return result

def main():
    print("🚀 Minimal Overhead Consciousness Vision")
    print("="*50)
    print("Optimized for Jetson - avoiding GPU transfer overhead")
    
    vision = MinimalGPUVision()
    
    # Camera pipeline - keep NVMM as long as possible
    gst_pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    print("\nRunning optimized version...")
    print("This should achieve 20+ FPS by avoiding GPU memory transfers")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = vision.process_frame(frame)
        
        cv2.imshow('Minimal GPU Vision', result)
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Average FPS: {fps:.1f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()