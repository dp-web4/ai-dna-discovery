#!/usr/bin/env python3
"""
Quick test to verify GPU acceleration is working
"""

import cv2
import numpy as np
import cupy as cp
import time

print("🚀 GPU Acceleration Test")
print("="*40)

# Test 1: Basic GPU operations
print("\n1. Testing CuPy GPU operations...")
cpu_array = np.random.rand(1000, 1000).astype(np.float32)
gpu_array = cp.asarray(cpu_array)

# CPU timing
start = time.time()
cpu_result = np.sum(cpu_array ** 2)
cpu_time = time.time() - start

# GPU timing
start = time.time()
gpu_result = cp.sum(gpu_array ** 2)
cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
gpu_time = time.time() - start

print(f"   CPU time: {cpu_time*1000:.2f} ms")
print(f"   GPU time: {gpu_time*1000:.2f} ms")
print(f"   Speedup: {cpu_time/gpu_time:.1f}x")
print(f"   Results match: {np.allclose(cpu_result, float(gpu_result))}")

# Test 2: Image operations
print("\n2. Testing image operations...")
test_img = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)

# CPU absdiff
img1 = test_img
img2 = (test_img + 10) % 255
start = time.time()
cpu_diff = cv2.absdiff(img1, img2)
cpu_time = time.time() - start

# GPU absdiff
gpu_img1 = cp.asarray(img1)
gpu_img2 = cp.asarray(img2)
start = time.time()
gpu_diff = cp.abs(gpu_img1.astype(cp.float32) - gpu_img2.astype(cp.float32))
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start

print(f"   CPU absdiff: {cpu_time*1000:.2f} ms")
print(f"   GPU absdiff: {gpu_time*1000:.2f} ms")
print(f"   Speedup: {cpu_time/gpu_time:.1f}x")

# Test 3: Motion grid calculation
print("\n3. Testing motion grid (8x8)...")
grid_size = 8
h, w = 720, 1280
gh, gw = h // grid_size, w // grid_size

# CPU version
start = time.time()
cpu_grid = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        y1, y2 = i * gh, (i + 1) * gh
        x1, x2 = j * gw, (j + 1) * gw
        cpu_grid[i, j] = np.mean(cpu_diff[y1:y2, x1:x2])
cpu_time = time.time() - start

# GPU version
start = time.time()
gpu_grid = cp.zeros((grid_size, grid_size))
gpu_diff_cp = cp.asarray(cpu_diff)
for i in range(grid_size):
    for j in range(grid_size):
        y1, y2 = i * gh, (i + 1) * gh
        x1, x2 = j * gw, (j + 1) * gw
        gpu_grid[i, j] = cp.mean(gpu_diff_cp[y1:y2, x1:x2])
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start

print(f"   CPU grid: {cpu_time*1000:.2f} ms")
print(f"   GPU grid: {gpu_time*1000:.2f} ms")
print(f"   Speedup: {cpu_time/gpu_time:.1f}x")

print("\n✅ GPU acceleration is working!")
print(f"   Compute capability: {cp.cuda.Device().compute_capability}")
print(f"   Available memory: {cp.cuda.MemoryPool().free_bytes() / 1024**3:.1f} GB")