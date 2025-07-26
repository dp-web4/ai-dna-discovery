#!/usr/bin/env python3
"""
Optimized GPU test - demonstrates when GPU is actually faster
"""

import cv2
import numpy as np
import cupy as cp
import time

print("🚀 Optimized GPU Acceleration Test")
print("="*50)
print("Testing with larger data sizes where GPU shines...")

# Test different data sizes
sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000), (4000, 4000)]

print("\n📊 Matrix multiplication test (A @ B):")
print("Size        CPU (ms)   GPU (ms)   Speedup")
print("-" * 45)

for size in sizes:
    # Create random matrices
    a_cpu = np.random.rand(size[0], size[1]).astype(np.float32)
    b_cpu = np.random.rand(size[1], size[0]).astype(np.float32)
    
    # CPU timing
    start = time.time()
    c_cpu = a_cpu @ b_cpu
    cpu_time = (time.time() - start) * 1000
    
    # GPU timing (including transfer time)
    start = time.time()
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    c_gpu = a_gpu @ b_gpu
    c_result = c_gpu.get()  # Transfer back
    gpu_time = (time.time() - start) * 1000
    
    speedup = cpu_time / gpu_time
    print(f"{size[0]}x{size[1]}    {cpu_time:8.2f}   {gpu_time:8.2f}   {speedup:6.2f}x")

# Test keeping data on GPU
print("\n🔥 Keeping data on GPU (multiple operations):")
print("Testing 50 sequential operations on 1000x1000 matrix...")

# Large matrix
size = 1000
data = np.random.rand(size, size).astype(np.float32)

# CPU: 50 operations
start = time.time()
result_cpu = data.copy()
for _ in range(50):
    result_cpu = result_cpu * 1.01 + 0.1
    result_cpu = np.sqrt(result_cpu)
cpu_time = (time.time() - start) * 1000

# GPU: 50 operations (data stays on GPU)
start = time.time()
data_gpu = cp.asarray(data)
result_gpu = data_gpu.copy()
for _ in range(50):
    result_gpu = result_gpu * 1.01 + 0.1
    result_gpu = cp.sqrt(result_gpu)
result_final = result_gpu.get()  # Only transfer at end
gpu_time = (time.time() - start) * 1000

print(f"CPU time: {cpu_time:.2f} ms")
print(f"GPU time: {gpu_time:.2f} ms")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")

# Vision-specific test
print("\n👁️ Vision pipeline test (1920x1080 image):")
img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

# CPU pipeline
start = time.time()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (21, 21), 0)
edges = cv2.Canny(blurred, 50, 150)
cpu_time = (time.time() - start) * 1000

# GPU pipeline (simplified)
start = time.time()
img_gpu = cp.asarray(img)
# Simple GPU grayscale conversion
gray_gpu = cp.mean(img_gpu, axis=2).astype(cp.uint8)
# Note: Full GPU pipeline would need custom kernels or VPI
gray_result = gray_gpu.get()
gpu_time = (time.time() - start) * 1000

print(f"CPU pipeline: {cpu_time:.2f} ms")
print(f"GPU transfer + grayscale: {gpu_time:.2f} ms")
print("\nNote: Full GPU vision pipeline requires custom CUDA kernels")
print("      or libraries like VPI/NPP for maximum performance")

print("\n💡 Key insights:")
print("1. GPU excels with large data (>500x500)")
print("2. Keep data on GPU for multiple operations")
print("3. Memory transfer is the bottleneck for small operations")
print("4. Vision needs specialized GPU libraries (VPI, NPP, TensorRT)")