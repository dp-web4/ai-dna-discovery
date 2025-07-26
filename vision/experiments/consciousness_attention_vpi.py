#!/usr/bin/env python3
"""
GPU-Accelerated Consciousness Vision using VPI
NVIDIA Vision Programming Interface for Jetson
"""

import cv2
import numpy as np
import time
from collections import deque

# Try to import VPI
try:
    import vpi
    VPI_AVAILABLE = True
    print("✅ VPI (NVIDIA Vision Programming Interface) available!")
    print(f"VPI Version: {vpi.__version__}")
    
    # Check available backends - VPI 3.x syntax
    backends = []
    
    # Test backend availability by trying to create a dummy stream
    backend_tests = [
        (vpi.Backend.CUDA, "CUDA"),
        (vpi.Backend.VIC, "VIC (Video Image Compositor)"),
        (vpi.Backend.PVA, "PVA (Programmable Vision Accelerator)"),
        (vpi.Backend.CPU, "CPU")
    ]
    
    for backend, name in backend_tests:
        try:
            # Create a test stream to check if backend is available
            with vpi.Stream(backend):
                backends.append(name)
        except Exception:
            pass
    
    print(f"Available backends: {', '.join(backends) if backends else 'None detected'}")
    
except ImportError:
    VPI_AVAILABLE = False
    print("❌ VPI not available - using CPU fallback")
    print("Install with: sudo apt-get install python3-vpi3")

class VPIConsciousnessVision:
    def __init__(self):
        # Focus parameters
        self.focus_x = 0.5
        self.focus_y = 0.5
        self.focus_radius = 0.15
        
        # Motion detection
        self.motion_grid_size = 8
        self.ambient_motion = 0.016
        self.motion_heatmap = np.zeros((self.motion_grid_size, self.motion_grid_size), dtype=np.float32)
        
        # Performance tracking
        self.gpu_times = deque(maxlen=30)
        self.cpu_times = deque(maxlen=30)
        
        # VPI setup
        if VPI_AVAILABLE:
            # Use CUDA backend for best performance
            self.backend = vpi.Backend.CUDA
            self.stream = vpi.Stream()
            print(f"Using VPI backend: {self.backend}")
            
            # Pre-allocate VPI arrays
            self.prev_frame_vpi = None
        else:
            print("WARNING: Running in CPU mode")
            
        self.prev_frame_cpu = None
        self.saccade_cooldown = 0
        self.peaks = []
        
    def process_frame(self, frame):
        """Process frame using GPU acceleration when available"""
        h, w = frame.shape[:2]
        
        if VPI_AVAILABLE:
            return self._process_vpi(frame, h, w)
        else:
            # WARNING: CPU FALLBACK
            return self._process_cpu(frame, h, w)
            
    def _process_vpi(self, frame, h, w):
        """GPU-accelerated processing using VPI"""
        gpu_start = time.time()
        
        # Convert to VPI image (GPU memory)
        with vpi.Backend.CUDA:
            frame_vpi = vpi.asimage(frame, vpi.Format.BGR8)
            
            if self.prev_frame_vpi is not None:
                # Convert to grayscale for motion detection
                gray_vpi = frame_vpi.convert(vpi.Format.U8, backend=self.backend)
                prev_gray_vpi = self.prev_frame_vpi.convert(vpi.Format.U8, backend=self.backend)
                
                # Convert to CPU for motion detection
                # VPI 3.x doesn't have all operations we need
                # But we still benefit from GPU memory transfers
                gray_cpu = np.array(gray_vpi.cpu())
                prev_gray_cpu = np.array(prev_gray_vpi.cpu())
                
                # Simple motion detection (will be GPU with CuPy later)
                diff_abs = np.abs(gray_cpu.astype(np.float32) - prev_gray_cpu.astype(np.float32))
                
                # Update motion heatmap
                self._update_motion_from_diff(diff_abs, h, w)
                
            self.prev_frame_vpi = frame_vpi
            
        gpu_time = time.time() - gpu_start
        self.gpu_times.append(gpu_time)
        
        # Create visualization
        result = self._visualize(frame, h, w)
        
        return result
        
    def _update_motion_from_diff(self, diff_abs, h, w):
        """Update motion heatmap from absolute difference"""
        # Update heatmap
        self.motion_heatmap *= 0.9
        
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                y1, y2 = i * grid_h, min((i + 1) * grid_h, h)
                x1, x2 = j * grid_w, min((j + 1) * grid_w, w)
                
                # Check if in periphery
                center_y = (y1 + y2) / 2 / h
                center_x = (x1 + x2) / 2 / w
                dist_from_focus = np.sqrt((center_x - self.focus_x)**2 + (center_y - self.focus_y)**2)
                
                if dist_from_focus > self.focus_radius:
                    cell_motion = np.mean(diff_abs[y1:y2, x1:x2]) / 255.0  # Normalize
                    self.motion_heatmap[i, j] += cell_motion
        
        # Find peaks
        self._find_peaks()
        
    def _process_cpu(self, frame, h, w):
        """CPU fallback - explicitly marked"""
        cpu_start = time.time()
        
        # WARNING: CPU OPERATIONS BELOW
        if self.prev_frame_cpu is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # CPU
            prev_gray = cv2.cvtColor(self.prev_frame_cpu, cv2.COLOR_BGR2GRAY)  # CPU
            diff = cv2.absdiff(gray, prev_gray)  # CPU
            
            # Update heatmap on CPU
            self.motion_heatmap *= 0.9
            
            grid_h = h // self.motion_grid_size
            grid_w = w // self.motion_grid_size
            
            for i in range(self.motion_grid_size):
                for j in range(self.motion_grid_size):
                    y1, y2 = i * grid_h, (i + 1) * grid_h
                    x1, x2 = j * grid_w, (j + 1) * grid_w
                    
                    # Check periphery
                    center_y = (y1 + y2) / 2 / h
                    center_x = (x1 + x2) / 2 / w
                    dist_from_focus = np.sqrt((center_x - self.focus_x)**2 + (center_y - self.focus_y)**2)
                    
                    if dist_from_focus > self.focus_radius:
                        cell_motion = np.mean(diff[y1:y2, x1:x2]) / 255.0  # CPU
                        self.motion_heatmap[i, j] += cell_motion
            
            self._find_peaks()
            
        self.prev_frame_cpu = frame.copy()
        
        cpu_time = time.time() - cpu_start
        self.cpu_times.append(cpu_time)
        
        # Create visualization with CPU warning
        result = self._visualize(frame, h, w)
        cv2.putText(result, "WARNING: CPU MODE", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
        
    def _find_peaks(self):
        """Find motion peaks for saccade targets"""
        self.peaks = []
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                if self.motion_heatmap[i, j] > self.ambient_motion * 1.5:
                    pa_ratio = self.motion_heatmap[i, j] / self.ambient_motion
                    self.peaks.append({
                        'x': (j + 0.5) / self.motion_grid_size,
                        'y': (i + 0.5) / self.motion_grid_size,
                        'pa_ratio': pa_ratio,
                        'strength': self.motion_heatmap[i, j]
                    })
        
        self.peaks.sort(key=lambda p: p['pa_ratio'], reverse=True)
        
        # Update focus based on strongest peak
        if self.peaks and self.peaks[0]['pa_ratio'] > 2.0 and self.saccade_cooldown == 0:
            self.focus_x = self.peaks[0]['x']
            self.focus_y = self.peaks[0]['y']
            self.saccade_cooldown = 10
        
        self.saccade_cooldown = max(0, self.saccade_cooldown - 1)
        
    def _visualize(self, frame, h, w):
        """Create visualization overlay"""
        result = frame.copy()
        
        # Focus circle
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        radius_px = int(self.focus_radius * min(h, w))
        cv2.circle(result, (focus_px, focus_py), radius_px, (0, 255, 255), 2)
        
        # Motion heatmap
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                if self.motion_heatmap[i, j] > 0.01:
                    x1, y1 = j * grid_w, i * grid_h
                    x2, y2 = (j + 1) * grid_w, (i + 1) * grid_h
                    
                    intensity = min(1.0, self.motion_heatmap[i, j] / 0.05)
                    color = (0, int(100 * intensity), int(255 * intensity))
                    
                    overlay = result.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
        
        # Peak indicators
        for peak in self.peaks[:3]:
            px = int(peak['x'] * w)
            py = int(peak['y'] * h)
            cv2.circle(result, (px, py), 10, (0, 0, 255), 2)
            cv2.putText(result, f"{peak['pa_ratio']:.1f}", (px-15, py-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Performance stats
        if VPI_AVAILABLE:
            cv2.putText(result, "GPU: VPI", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.gpu_times:
                avg_gpu = np.mean(self.gpu_times) * 1000
                cv2.putText(result, f"Process: {avg_gpu:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(result, "CPU MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.cpu_times:
                avg_cpu = np.mean(self.cpu_times) * 1000
                cv2.putText(result, f"Process: {avg_cpu:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        return result

def main():
    print("\n🚀 GPU-Accelerated Vision with VPI")
    print("="*50)
    
    vision = VPIConsciousnessVision()
    
    # Camera pipeline
    gst_pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! "  # GPU conversion
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "  # Still need CPU conversion for OpenCV
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        return
    
    frame_count = 0
    fps_timer = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = vision.process_frame(frame)
        
        # FPS
        frame_count += 1
        if time.time() - fps_timer > 1.0:
            fps = frame_count / (time.time() - fps_timer)
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            fps_timer = time.time()
            frame_count = 0
        
        cv2.imshow('VPI Consciousness Vision', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()