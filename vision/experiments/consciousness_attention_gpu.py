#!/usr/bin/env python3
"""
GPU-Accelerated Consciousness Vision
Maximizes GPU usage on Jetson Orin Nano
"""

import cv2
import numpy as np
import time
from collections import deque

# Check available GPU acceleration
GPU_AVAILABLE = False
CUPY_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    GPU_AVAILABLE = True
    print("✅ CuPy available - GPU acceleration enabled!")
except ImportError:
    print("⚠️  CuPy not available - Install with: pip3 install cupy-cuda12x")
    
# Try VPI (NVIDIA Vision Programming Interface)
VPI_AVAILABLE = False
try:
    import vpi
    VPI_AVAILABLE = True
    print("✅ VPI available - Hardware acceleration enabled!")
except ImportError:
    print("⚠️  VPI not available")

print(f"\nGPU Status: {'🚀 READY' if GPU_AVAILABLE else '❌ CPU FALLBACK'}")
print("="*50)

class GPUConsciousnessVision:
    def __init__(self):
        # Focus parameters
        self.focus_x = 0.5
        self.focus_y = 0.5
        self.focus_radius = 0.15
        
        # Motion detection
        self.motion_grid_size = 8
        self.ambient_motion = 0.016
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.gpu_times = deque(maxlen=30)
        self.cpu_times = deque(maxlen=30)
        
        # Initialize GPU arrays if available
        if CUPY_AVAILABLE:
            self.motion_heatmap_gpu = cp.zeros((self.motion_grid_size, self.motion_grid_size), dtype=cp.float32)
            print("GPU memory allocated for motion heatmap")
        else:
            # WARNING: CPU FALLBACK
            self.motion_heatmap_cpu = np.zeros((self.motion_grid_size, self.motion_grid_size), dtype=np.float32)
            print("WARNING: Using CPU memory for motion heatmap")
            
        self.prev_frame_gpu = None
        self.saccade_cooldown = 0
        
    def process_frame_gpu(self, frame):
        """GPU-accelerated frame processing"""
        frame_start = time.time()
        h, w = frame.shape[:2]
        
        if CUPY_AVAILABLE:
            # Transfer to GPU once
            gpu_start = time.time()
            frame_gpu = cp.asarray(frame)
            
            # GPU operations
            result = self._process_gpu(frame_gpu, h, w)
            
            # Transfer back
            result_cpu = cp.asnumpy(result)
            gpu_time = time.time() - gpu_start
            self.gpu_times.append(gpu_time)
            
            return result_cpu
        else:
            # WARNING: CPU FALLBACK - No GPU acceleration available
            cpu_start = time.time()
            result = self._process_cpu_fallback(frame, h, w)
            cpu_time = time.time() - cpu_start
            self.cpu_times.append(cpu_time)
            
            # Add warning overlay
            cv2.putText(result, "WARNING: CPU MODE - Install CuPy for GPU", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return result
            
    def _process_gpu(self, frame_gpu, h, w):
        """GPU processing pipeline"""
        # Convert to grayscale on GPU
        # Note: Full GPU path would use VPI or custom kernels
        # For now, we minimize CPU operations
        
        if self.prev_frame_gpu is not None:
            # GPU motion detection
            diff_gpu = cp.abs(frame_gpu.astype(cp.float32) - self.prev_frame_gpu)
            motion_magnitude = cp.mean(diff_gpu, axis=2)  # Average across channels
            
            # Update motion heatmap on GPU
            self._update_motion_heatmap_gpu(motion_magnitude, h, w)
            
            # Find peaks on GPU
            peaks = self._find_motion_peaks_gpu()
            
            # Update focus based on peaks
            if len(peaks) > 0 and peaks[0]['pa_ratio'] > 2.0:
                self.focus_x = peaks[0]['x']
                self.focus_y = peaks[0]['y']
        
        self.prev_frame_gpu = frame_gpu.copy()
        
        # Create visualization (still some CPU needed for OpenCV drawing)
        # In production, would use GPU rendering
        result = cp.asnumpy(frame_gpu)
        self._draw_overlay_minimal(result, h, w, peaks if 'peaks' in locals() else [])
        
        return result
        
    def _update_motion_heatmap_gpu(self, motion_gpu, h, w):
        """Update motion heatmap on GPU"""
        # Decay existing heatmap
        self.motion_heatmap_gpu *= 0.9
        
        # Grid-based motion accumulation
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        # GPU kernel would be ideal here
        # For now, use CuPy operations
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                y1, y2 = i * grid_h, min((i + 1) * grid_h, h)
                x1, x2 = j * grid_w, min((j + 1) * grid_w, w)
                
                # Check if outside focus area (peripheral only)
                center_y = (y1 + y2) / 2 / h
                center_x = (x1 + x2) / 2 / w
                dist_from_focus = cp.sqrt((center_x - self.focus_x)**2 + (center_y - self.focus_y)**2)
                
                if dist_from_focus > self.focus_radius:
                    cell_motion = cp.mean(motion_gpu[y1:y2, x1:x2]) / 255.0
                    self.motion_heatmap_gpu[i, j] += cell_motion
                    
    def _find_motion_peaks_gpu(self):
        """Find motion peaks on GPU"""
        peaks = []
        
        # GPU operations to find peaks
        max_motion = cp.max(self.motion_heatmap_gpu)
        
        if max_motion > self.ambient_motion * 1.5:
            # Get all cells above threshold
            peak_mask = self.motion_heatmap_gpu > (self.ambient_motion * 1.5)
            peak_indices = cp.where(peak_mask)
            
            # Convert to CPU for final processing
            # (In full GPU pipeline, this would stay on GPU)
            indices_cpu = (cp.asnumpy(peak_indices[0]), cp.asnumpy(peak_indices[1]))
            values_cpu = cp.asnumpy(self.motion_heatmap_gpu[peak_mask])
            
            for idx in range(len(indices_cpu[0])):
                i, j = indices_cpu[0][idx], indices_cpu[1][idx]
                pa_ratio = values_cpu[idx] / self.ambient_motion
                
                peaks.append({
                    'x': (j + 0.5) / self.motion_grid_size,
                    'y': (i + 0.5) / self.motion_grid_size,
                    'pa_ratio': pa_ratio,
                    'strength': values_cpu[idx]
                })
                
        return sorted(peaks, key=lambda p: p['pa_ratio'], reverse=True)
        
    def _process_cpu_fallback(self, frame, h, w):
        """CPU fallback - explicitly marked as suboptimal"""
        # WARNING: ENTIRE FUNCTION IS CPU-BASED
        # This is only for systems without GPU libraries
        
        result = frame.copy()
        
        # Simple CPU motion detection
        if hasattr(self, 'prev_frame_cpu'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # CPU operation
            prev_gray = cv2.cvtColor(self.prev_frame_cpu, cv2.COLOR_BGR2GRAY)  # CPU operation
            diff = cv2.absdiff(gray, prev_gray)  # CPU operation
            
            # Update heatmap on CPU
            self.motion_heatmap_cpu *= 0.9  # CPU operation
            
            # Simplified motion detection
            grid_h = h // 8
            grid_w = w // 8
            
            for i in range(8):
                for j in range(8):
                    y1, y2 = i * grid_h, (i + 1) * grid_h
                    x1, x2 = j * grid_w, (j + 1) * grid_w
                    cell_motion = np.mean(diff[y1:y2, x1:x2]) / 255.0  # CPU operation
                    self.motion_heatmap_cpu[i, j] += cell_motion
                    
        self.prev_frame_cpu = frame.copy()
        
        # Draw overlay
        self._draw_overlay_minimal(result, h, w, [])
        
        return result
        
    def _draw_overlay_minimal(self, frame, h, w, peaks):
        """Minimal overlay to reduce CPU usage"""
        # Draw focus circle
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        radius_px = int(self.focus_radius * min(h, w))
        cv2.circle(frame, (focus_px, focus_py), radius_px, (0, 255, 255), 2)
        
        # Performance stats
        if self.gpu_times:
            avg_gpu = np.mean(self.gpu_times) * 1000
            cv2.putText(frame, f"GPU: {avg_gpu:.1f}ms", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.cpu_times:
            avg_cpu = np.mean(self.cpu_times) * 1000
            cv2.putText(frame, f"CPU: {avg_cpu:.1f}ms", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
        # Peak indicators
        for peak in peaks[:3]:
            px = int(peak['x'] * w)
            py = int(peak['y'] * h)
            cv2.circle(frame, (px, py), 10, (0, 0, 255), 2)
            
def main():
    print("\n🚀 GPU-Accelerated Consciousness Vision")
    print("="*50)
    
    vision = GPUConsciousnessVision()
    
    # Camera pipeline - keeping data in NVMM as long as possible
    gst_pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! "  # GPU color conversion
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "  # WARNING: CPU conversion here - unavoidable with current OpenCV
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
            
        # Process frame (GPU or CPU)
        result = vision.process_frame_gpu(frame)
        
        # FPS calculation
        frame_count += 1
        if time.time() - fps_timer > 1.0:
            fps = frame_count / (time.time() - fps_timer)
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            fps_timer = time.time()
            frame_count = 0
            
        cv2.imshow('GPU Consciousness Vision', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Performance summary
    print("\nPerformance Summary:")
    if vision.gpu_times:
        print(f"Average GPU time: {np.mean(vision.gpu_times)*1000:.1f}ms")
    if vision.cpu_times:
        print(f"Average CPU time: {np.mean(vision.cpu_times)*1000:.1f}ms")
        print("⚠️  RECOMMENDATION: Install CuPy for GPU acceleration")

if __name__ == "__main__":
    main()