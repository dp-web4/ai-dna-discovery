#!/usr/bin/env python3
"""
Debug GPU performance issues
"""

import time
import numpy as np
import cupy as cp
import cv2

print("🔍 GPU Performance Debugging")
print("="*50)

# 1. Check GPU memory
mempool = cp.get_default_memory_pool()
print(f"GPU Memory used: {mempool.used_bytes() / 1024**2:.1f} MB")
print(f"GPU Memory total: {mempool.total_bytes() / 1024**2:.1f} MB")

# 2. Test simple GPU operations
print("\n📊 Testing GPU operations...")

# Small operation
x = cp.random.rand(100, 100)
start = time.time()
for _ in range(1000):
    y = x * 2.0
cp.cuda.Stream.null.synchronize()
print(f"1000 small ops: {(time.time()-start)*1000:.1f}ms")

# Memory transfer test
print("\n🔄 Testing memory transfers...")
cpu_data = np.random.rand(720, 1280, 3).astype(np.float32)

# CPU to GPU
start = time.time()
gpu_data = cp.asarray(cpu_data)
cp.cuda.Stream.null.synchronize()
print(f"CPU->GPU (720x1280x3): {(time.time()-start)*1000:.1f}ms")

# GPU to CPU
start = time.time()
cpu_back = cp.asnumpy(gpu_data)
print(f"GPU->CPU (720x1280x3): {(time.time()-start)*1000:.1f}ms")

# 3. Test video frame processing
print("\n🎥 Testing frame processing...")

frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Pure CPU
start = time.time()
gray_cpu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
diff_cpu = cv2.absdiff(gray_cpu, gray_cpu)
cpu_time = (time.time() - start) * 1000

# GPU with transfers
start = time.time()
frame_gpu = cp.asarray(frame)
gray_gpu = cp.mean(frame_gpu, axis=2).astype(cp.uint8)
diff_gpu = cp.abs(gray_gpu.astype(cp.float32) - gray_gpu.astype(cp.float32))
result = cp.asnumpy(diff_gpu)
gpu_time = (time.time() - start) * 1000

print(f"CPU processing: {cpu_time:.1f}ms")
print(f"GPU processing (with transfers): {gpu_time:.1f}ms")

# 4. Check for throttling
print("\n🌡️ Checking for throttling...")
import subprocess
try:
    result = subprocess.run(['tegrastats', '--interval', '1000'], 
                          capture_output=True, text=True, timeout=2)
    print("Tegrastats output:", result.stdout)
except:
    print("Could not run tegrastats")

# 5. Test continuous operations
print("\n⏱️ Testing continuous GPU operations...")
frame_gpu = cp.asarray(frame)
times = []

for i in range(30):
    start = time.time()
    gray = cp.mean(frame_gpu, axis=2).astype(cp.uint8)
    _ = cp.abs(gray.astype(cp.float32) - gray.astype(cp.float32))
    cp.cuda.Stream.null.synchronize()
    times.append((time.time() - start) * 1000)

print(f"First frame: {times[0]:.1f}ms")
print(f"Average (frames 10-30): {np.mean(times[10:]):.1f}ms")
print(f"Min time: {min(times):.1f}ms")

print("\n💡 Diagnosis:")
if times[0] > 100:
    print("❌ GPU initialization is very slow")
if np.mean(times[10:]) > 50:
    print("❌ GPU operations are slower than expected")
if gpu_time > cpu_time * 2:
    print("❌ Memory transfer overhead is dominating")
    
print("\n🔧 Recommendations:")
print("1. Check power mode: sudo nvpmodel -q")
print("2. Set max performance: sudo nvpmodel -m 0")
print("3. Check thermal: cat /sys/class/thermal/thermal_zone*/temp")
print("4. Monitor with: sudo jtop")