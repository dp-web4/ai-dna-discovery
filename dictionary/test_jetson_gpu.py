#!/usr/bin/env python3
"""
Quick test script to verify Jetson GPU capabilities
"""

import torch
import subprocess

print("=== Jetson GPU Test ===\n")

# Check PyTorch CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    props = torch.cuda.get_device_properties(0)
    print(f"\nMemory Info:")
    print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"  Memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    # Compute capability
    print(f"\nCompute capability: {props.major}.{props.minor}")
    print(f"Multi-processor count: {props.multi_processor_count}")
    
    # Quick tensor test
    print("\nTensor test:")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"  Matrix multiplication successful: {z.shape}")
    
    # Cleanup
    del x, y, z
    torch.cuda.empty_cache()
else:
    print("CUDA is not available!")
    print("This might be a driver or PyTorch installation issue.")

# System info
print("\n=== System Info ===")
try:
    # Jetson-specific info
    result = subprocess.run(['jetson_release'], capture_output=True, text=True)
    if result.returncode == 0:
        print("Jetson Info:")
        print(result.stdout)
except:
    print("jetson_release command not found")

# Try nvidia-smi
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\nnvidia-smi output:")
        print(result.stdout)
except:
    print("nvidia-smi not available")

print("\n✅ Test complete!")