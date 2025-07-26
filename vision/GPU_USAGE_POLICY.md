# GPU USAGE POLICY FOR JETSON VISION EXPERIMENTS

## CRITICAL DIRECTIVE
**ALWAYS USE GPU AS MUCH AS POSSIBLE**

The Jetson Orin Nano has:
- 1024 CUDA cores
- 32 Tensor cores  
- 40 TOPS AI performance
- Hardware video encode/decode

Using CPU for vision processing is wasting this power!

## GPU First Principles

### 1. Keep Data on GPU
- Start with NVMM buffers from camera
- Process on GPU
- Only transfer to CPU when absolutely necessary

### 2. Explicit CPU Usage Notification
When CPU is used, MUST note:
```python
# WARNING: CPU OPERATION - Consider GPU alternative
diff = cv2.absdiff(frame1, frame2)  # Should use cv2.cuda.absdiff
```

### 3. GPU Alternatives Checklist
| CPU Operation | GPU Alternative |
|--------------|----------------|
| cv2.absdiff() | cv2.cuda.absdiff() |
| cv2.threshold() | cv2.cuda.threshold() |
| cv2.GaussianBlur() | cv2.cuda.GaussianBlur() |
| numpy operations | cupy operations |
| for loops | CUDA kernels |
| cv2.resize() | cv2.cuda.resize() |

### 4. Memory Transfer Awareness
```python
# BAD: Multiple transfers
frame = cap.read()          # GPU->CPU
processed = process(frame)  # CPU
display(processed)          # CPU->GPU

# GOOD: Stay on GPU
gpu_frame = cap.read_gpu()  # Stays on GPU
gpu_processed = process_gpu(gpu_frame)
display_gpu(gpu_processed)
```

### 5. Tools to Use
- **OpenCV CUDA**: cv2.cuda.* functions
- **CuPy**: GPU numpy replacement
- **VPI**: NVIDIA Vision Programming Interface
- **TensorRT**: Neural network inference
- **Custom CUDA**: For unique operations

## Implementation Checklist
Before committing any vision code:
- [ ] Is camera data kept in NVMM/GPU memory?
- [ ] Are all operations GPU-accelerated?
- [ ] Are CPU operations explicitly marked with warnings?
- [ ] Is memory transfer minimized?
- [ ] Could custom CUDA kernels help?

## Performance Targets
- Motion detection: <5ms per frame (GPU)
- Edge detection: <3ms per frame (GPU)
- Full pipeline: 60+ FPS

Remember: The Jetson's strength is GPU processing. Every CPU operation is a missed opportunity!