#!/usr/bin/env python3
"""
Optimized GPU-Accelerated Consciousness Vision
Now with actual GPU performance!
"""

import cv2
import numpy as np
import time
from collections import deque

# GPU acceleration with CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy available - GPU acceleration enabled!")
    print(f"   CUDA compute capability: {cp.cuda.Device().compute_capability}")
    # Warm up GPU
    _ = cp.zeros((100, 100))
    cp.cuda.Stream.null.synchronize()
    print("   GPU warmed up and ready!")
except ImportError:
    CUPY_AVAILABLE = False
    print("❌ CuPy not available - CPU fallback")

class OptimizedGPUVision:
    def __init__(self):
        # Focus parameters
        self.focus_x = 0.5
        self.focus_y = 0.5
        self.focus_radius = 0.15
        self.focus_radius_target = 0.15
        
        # Motion detection
        self.motion_grid_size = 8
        self.ambient_motion = 0.016
        
        # Performance tracking
        self.gpu_times = deque(maxlen=30)
        self.fps_history = deque(maxlen=30)
        
        if CUPY_AVAILABLE:
            # Pre-allocate GPU memory
            self.motion_heatmap = cp.zeros((self.motion_grid_size, self.motion_grid_size), dtype=cp.float32)
            self.prev_gray_gpu = None
            # Create dedicated stream for async operations
            self.stream = cp.cuda.Stream()
            print("✅ GPU memory pre-allocated")
        else:
            self.motion_heatmap = np.zeros((self.motion_grid_size, self.motion_grid_size), dtype=np.float32)
            self.prev_gray_cpu = None
            
        self.saccade_cooldown = 0
        self.peaks = []
        
    def process_frame(self, frame):
        """Main processing with GPU optimization"""
        h, w = frame.shape[:2]
        
        if CUPY_AVAILABLE:
            return self._process_gpu(frame, h, w)
        else:
            return self._process_cpu(frame, h, w)
            
    def _process_gpu(self, frame, h, w):
        """Optimized GPU processing"""
        gpu_start = time.time()
        
        with self.stream:
            # Transfer to GPU and convert to grayscale in one operation
            frame_gpu = cp.asarray(frame)
            gray_gpu = cp.mean(frame_gpu, axis=2).astype(cp.uint8)
            
            if self.prev_gray_gpu is not None:
                # Motion detection on GPU
                diff = cp.abs(gray_gpu.astype(cp.float32) - self.prev_gray_gpu.astype(cp.float32))
                
                # Update motion heatmap
                self._update_motion_gpu(diff, h, w)
                
            self.prev_gray_gpu = gray_gpu
        
        # Synchronize and get results
        self.stream.synchronize()
        
        # Find peaks (small data, OK to transfer)
        self._find_peaks_gpu()
        
        # Update focus
        self._update_focus()
        
        gpu_time = time.time() - gpu_start
        self.gpu_times.append(gpu_time)
        
        # Visualization (requires CPU)
        result = self._visualize_gpu(frame, h, w)
        
        return result
        
    def _update_motion_gpu(self, diff, h, w):
        """Update motion heatmap on GPU"""
        # Decay
        self.motion_heatmap *= 0.9
        
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        # Create focus mask on GPU
        y_coords = cp.arange(h).reshape(-1, 1) / h
        x_coords = cp.arange(w).reshape(1, -1) / w
        
        dist_from_focus = cp.sqrt(
            (x_coords - self.focus_x)**2 + (y_coords - self.focus_y)**2
        )
        periphery_mask = (dist_from_focus > self.focus_radius).astype(cp.float32)
        
        # Apply mask to diff
        masked_diff = diff * periphery_mask
        
        # Update grid (still uses loop but data stays on GPU)
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                y1, y2 = i * grid_h, min((i + 1) * grid_h, h)
                x1, x2 = j * grid_w, min((j + 1) * grid_w, w)
                
                cell_motion = cp.mean(masked_diff[y1:y2, x1:x2]) / 255.0
                self.motion_heatmap[i, j] += cell_motion
                
    def _find_peaks_gpu(self):
        """Find motion peaks"""
        self.peaks = []
        
        # Transfer small heatmap to CPU for peak finding
        heatmap_cpu = cp.asnumpy(self.motion_heatmap)
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                if heatmap_cpu[i, j] > self.ambient_motion * 1.5:
                    pa_ratio = heatmap_cpu[i, j] / self.ambient_motion
                    self.peaks.append({
                        'x': (j + 0.5) / self.motion_grid_size,
                        'y': (i + 0.5) / self.motion_grid_size,
                        'pa_ratio': pa_ratio,
                        'strength': heatmap_cpu[i, j]
                    })
        
        self.peaks.sort(key=lambda p: p['pa_ratio'], reverse=True)
        
    def _update_focus(self):
        """Update focus position based on motion"""
        self.saccade_cooldown = max(0, self.saccade_cooldown - 1)
        
        if self.peaks and self.saccade_cooldown == 0:
            peak = self.peaks[0]
            if peak['pa_ratio'] > 2.0:
                self.focus_x = peak['x']
                self.focus_y = peak['y']
                self.saccade_cooldown = 10
                self.focus_radius_target = 0.2
        
        # Smooth radius changes
        self.focus_radius += (self.focus_radius_target - self.focus_radius) * 0.1
        
    def _visualize_gpu(self, frame, h, w):
        """Create visualization"""
        result = frame.copy()
        
        # Draw focus circle
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        radius_px = int(self.focus_radius * min(h, w))
        cv2.circle(result, (focus_px, focus_py), radius_px, (0, 255, 255), 2)
        
        # Draw motion heatmap
        if CUPY_AVAILABLE:
            heatmap_cpu = cp.asnumpy(self.motion_heatmap)
        else:
            heatmap_cpu = self.motion_heatmap
            
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                if heatmap_cpu[i, j] > 0.01:
                    x1, y1 = j * grid_w, i * grid_h
                    x2, y2 = (j + 1) * grid_w, (i + 1) * grid_h
                    
                    intensity = min(1.0, heatmap_cpu[i, j] / 0.05)
                    color = (0, int(100 * intensity), int(255 * intensity))
                    
                    overlay = result.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
        
        # Performance stats
        if CUPY_AVAILABLE:
            cv2.putText(result, "GPU: CuPy CUDA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.gpu_times:
                avg_time = np.mean(self.gpu_times) * 1000
                cv2.putText(result, f"Process: {avg_time:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(result, "WARNING: CPU MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Peak indicators
        for peak in self.peaks[:3]:
            px = int(peak['x'] * w)
            py = int(peak['y'] * h)
            cv2.circle(result, (px, py), 10, (0, 0, 255), 2)
            cv2.putText(result, f"{peak['pa_ratio']:.1f}", (px-15, py-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return result
        
    def _process_cpu(self, frame, h, w):
        """CPU fallback"""
        # WARNING: CPU OPERATION
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if hasattr(self, 'prev_gray_cpu') and self.prev_gray_cpu is not None:
            diff = cv2.absdiff(gray, self.prev_gray_cpu)
            # Simplified CPU processing...
            
        self.prev_gray_cpu = gray
        return self._visualize_gpu(frame, h, w)

def main():
    print("\n🚀 Optimized GPU Consciousness Vision")
    print("="*50)
    
    vision = OptimizedGPUVision()
    
    # Camera pipeline
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
    
    frame_count = 0
    fps_timer = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = vision.process_frame(frame)
        
        # FPS calculation
        frame_count += 1
        if time.time() - fps_timer > 1.0:
            fps = frame_count / (time.time() - fps_timer)
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            vision.fps_history.append(fps)
            fps_timer = time.time()
            frame_count = 0
            
            # Print performance summary every second
            if CUPY_AVAILABLE and vision.gpu_times:
                avg_gpu = np.mean(vision.gpu_times) * 1000
                print(f"GPU: {avg_gpu:.1f}ms | FPS: {fps:.1f}")
        
        cv2.imshow('Optimized GPU Vision', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final performance summary
    if CUPY_AVAILABLE and vision.fps_history:
        print(f"\nPerformance Summary:")
        print(f"Average FPS: {np.mean(vision.fps_history):.1f}")
        print(f"GPU processing time: {np.mean(vision.gpu_times)*1000:.1f}ms")
        print("✅ GPU acceleration successful!")

if __name__ == "__main__":
    main()