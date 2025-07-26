#!/usr/bin/env python3
"""
GPU-Accelerated Edge Detection with CUDA
Demonstrates custom GPU kernels for vision processing
"""

import cv2
import numpy as np
import cupy as cp  # GPU arrays
import time

# Check if CuPy is available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available - GPU acceleration enabled!")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - install with: pip3 install cupy-cuda12x")

class GPUEdgeDetector:
    def __init__(self):
        self.sobel_x = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=np.float32)
        
        self.sobel_y = np.array([[-1, -2, -1],
                                  [ 0,  0,  0],
                                  [ 1,  2,  1]], dtype=np.float32)
        
        if CUPY_AVAILABLE:
            # Transfer kernels to GPU once
            self.sobel_x_gpu = cp.asarray(self.sobel_x)
            self.sobel_y_gpu = cp.asarray(self.sobel_y)
            
            # Compile custom CUDA kernel for edge detection
            self.edge_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void sobel_edge_detection(const unsigned char* input, 
                                     float* output,
                                     const int width, 
                                     const int height) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
                    float gx = 0.0f;
                    float gy = 0.0f;
                    
                    // Sobel X
                    gx += -1.0f * input[(y-1) * width + (x-1)];
                    gx += -2.0f * input[y * width + (x-1)];
                    gx += -1.0f * input[(y+1) * width + (x-1)];
                    gx +=  1.0f * input[(y-1) * width + (x+1)];
                    gx +=  2.0f * input[y * width + (x+1)];
                    gx +=  1.0f * input[(y+1) * width + (x+1)];
                    
                    // Sobel Y
                    gy += -1.0f * input[(y-1) * width + (x-1)];
                    gy += -2.0f * input[(y-1) * width + x];
                    gy += -1.0f * input[(y-1) * width + (x+1)];
                    gy +=  1.0f * input[(y+1) * width + (x-1)];
                    gy +=  2.0f * input[(y+1) * width + x];
                    gy +=  1.0f * input[(y+1) * width + (x+1)];
                    
                    // Magnitude
                    output[y * width + x] = sqrtf(gx * gx + gy * gy);
                }
            }
            ''', 'sobel_edge_detection')
    
    def detect_edges_gpu(self, frame):
        """GPU-accelerated edge detection"""
        if not CUPY_AVAILABLE:
            return self.detect_edges_cpu(frame)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Transfer to GPU
        gpu_input = cp.asarray(gray)
        gpu_output = cp.zeros_like(gpu_input, dtype=cp.float32)
        
        # Configure GPU grid
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (gray.shape[1] + threads_per_block[0] - 1) // threads_per_block[0],
            (gray.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
        )
        
        # Launch kernel
        self.edge_kernel(blocks_per_grid, threads_per_block,
                        (gpu_input, gpu_output, gray.shape[1], gray.shape[0]))
        
        # Transfer back to CPU
        edges = cp.asnumpy(gpu_output)
        
        # Normalize to 0-255
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        
        return edges
    
    def detect_edges_cpu(self, frame):
        """CPU fallback for edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sobel edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        edges = np.clip(magnitude, 0, 255).astype(np.uint8)
        
        return edges
    
    def create_artistic_edges(self, frame, edges):
        """Create artistic visualization of edges"""
        # Create colored edge map
        edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        
        # Blend with original
        result = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
        
        return result

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.gpu_times = []
        
    def start_frame(self):
        self.frame_start = time.time()
        
    def end_frame(self):
        frame_time = time.time() - self.frame_start
        self.frame_times.append(frame_time)
        
        # Keep last 30 frames
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            
    def get_fps(self):
        if not self.frame_times:
            return 0
        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0
    
    def draw_stats(self, frame, using_gpu=False):
        """Draw performance statistics on frame"""
        fps = self.get_fps()
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # GPU/CPU indicator
        mode = "GPU" if using_gpu else "CPU"
        color = (0, 255, 255) if using_gpu else (0, 165, 255)
        cv2.putText(frame, f"Mode: {mode}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Frame time
        if self.frame_times:
            avg_ms = np.mean(self.frame_times) * 1000
            cv2.putText(frame, f"Frame: {avg_ms:.1f}ms", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    print("⚡ GPU-Accelerated Edge Detection")
    print("=" * 40)
    print("Press 'q' to quit")
    print("Press 's' to save snapshot")
    print("Press 'a' to toggle artistic mode")
    print("=" * 40)
    
    # Initialize
    detector = GPUEdgeDetector()
    monitor = PerformanceMonitor()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    artistic_mode = True
    
    while True:
        monitor.start_frame()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect edges
        edges = detector.detect_edges_gpu(frame)
        
        # Create visualization
        if artistic_mode:
            result = detector.create_artistic_edges(frame, edges)
        else:
            # Just show edges
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
        # Draw stats
        monitor.draw_stats(result, using_gpu=CUPY_AVAILABLE)
        
        monitor.end_frame()
        
        # Display
        cv2.imshow('GPU Edge Detection', result)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            cv2.imwrite(f"edges_{timestamp}.jpg", result)
            cv2.imwrite(f"edges_raw_{timestamp}.jpg", edges)
            print(f"Saved snapshots")
        elif key == ord('a'):
            artistic_mode = not artistic_mode
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    if monitor.frame_times:
        print(f"\nAverage FPS: {monitor.get_fps():.1f}")
        print(f"Average frame time: {np.mean(monitor.frame_times)*1000:.1f}ms")

if __name__ == "__main__":
    main()