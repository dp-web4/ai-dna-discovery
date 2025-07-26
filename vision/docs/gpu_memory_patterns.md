# GPU Memory Buffer Patterns for Jetson Vision

## Overview
Efficient GPU memory management is crucial for real-time vision on Jetson. The key is minimizing memory copies between CPU and GPU.

## Memory Types on Jetson

### 1. NVMM (NVIDIA Multi-Media) Buffers
- Hardware-accelerated video memory
- Zero-copy between camera/encode/decode
- Accessed via GStreamer or Multimedia API
- Cannot be directly accessed by CPU

### 2. CUDA Unified Memory
- Accessible by both CPU and GPU
- Automatic migration between processors
- `cudaMallocManaged()` allocation
- Good for prototyping

### 3. CUDA Device Memory
- GPU-only memory
- Fastest for GPU operations
- Requires explicit copies to/from CPU
- `cudaMalloc()` allocation

### 4. Pinned Host Memory
- CPU memory that's page-locked
- Faster CPU↔GPU transfers
- `cudaMallocHost()` allocation

## Vision Pipeline Patterns

### Pattern 1: Zero-Copy Camera to Inference
```python
# GStreamer pipeline keeps data in GPU memory
pipeline = '''
    nvarguscamerasrc ! 
    video/x-raw(memory:NVMM), width=1920, height=1080 !
    nvvidconv ! 
    video/x-raw(memory:NVMM), format=NV12 !
    nvvidconv ! 
    video/x-raw, format=BGRx !
    videoconvert !
    video/x-raw, format=BGR !
    appsink
'''
```

### Pattern 2: Direct CUDA Processing
```cpp
// Allocate GPU memory for image
cudaMalloc(&d_input, width * height * channels);
cudaMalloc(&d_output, output_size);

// Process on GPU without CPU involvement
preprocessCUDA<<<blocks, threads>>>(d_input, d_processed);
inferenceEngine->enqueue(d_processed, d_output, stream);
postprocessCUDA<<<blocks, threads>>>(d_output, d_result);
```

### Pattern 3: Unified Memory for Flexibility
```cpp
// Allocate unified memory
cudaMallocManaged(&image, size);

// Can access from CPU
cv::Mat mat(height, width, CV_8UC3, image);

// And from GPU kernel
processImage<<<blocks, threads>>>(image, width, height);
cudaDeviceSynchronize();
```

### Pattern 4: Multi-Stream Processing
```cpp
// Create CUDA streams for parallel execution
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Process multiple frames concurrently
preprocessCUDA<<<blocks, threads, 0, stream1>>>(frame1);
preprocessCUDA<<<blocks, threads, 0, stream2>>>(frame2);

// Inference can overlap with preprocessing
engine->enqueueAsync(frame1, output1, stream1);
engine->enqueueAsync(frame2, output2, stream2);
```

## Best Practices

### 1. Use NVMM for Video Pipeline
```python
# Good: Stays in GPU memory
"nvarguscamerasrc ! video/x-raw(memory:NVMM) ! nvjpegenc ! ..."

# Bad: Copies to CPU
"nvarguscamerasrc ! video/x-raw ! jpegenc ! ..."
```

### 2. Batch Operations
```cpp
// Process multiple images at once
cudaMemcpyAsync(d_batch, h_images, batch_size * image_size, 
                cudaMemcpyHostToDevice, stream);
batchInference<<<grid, block, 0, stream>>>(d_batch, d_results, batch_size);
```

### 3. Overlap Compute and Transfer
```cpp
// Use streams to hide memory transfer latency
for (int i = 0; i < num_frames; i++) {
    int stream_id = i % num_streams;
    
    // Async copy next frame while processing current
    cudaMemcpyAsync(d_input[stream_id], h_input[i], size, 
                    cudaMemcpyHostToDevice, streams[stream_id]);
    
    // Process
    inference(d_input[stream_id], d_output[stream_id], streams[stream_id]);
    
    // Copy results back
    cudaMemcpyAsync(h_output[i], d_output[stream_id], output_size,
                    cudaMemcpyDeviceToHost, streams[stream_id]);
}
```

### 4. Memory Pooling
```cpp
// Reuse allocations instead of frequent malloc/free
class MemoryPool {
    std::queue<void*> available;
    size_t block_size;
    
    void* acquire() {
        if (available.empty()) {
            void* ptr;
            cudaMalloc(&ptr, block_size);
            return ptr;
        }
        void* ptr = available.front();
        available.pop();
        return ptr;
    }
    
    void release(void* ptr) {
        available.push(ptr);
    }
};
```

## Jetson-Specific Optimizations

### 1. Use Hardware Encoders/Decoders
- NVENC for H.264/H.265 encoding
- NVDEC for video decoding
- NVJPEG for JPEG encode/decode
- All work with NVMM buffers

### 2. Tensor Core Usage (Orin)
- Use FP16/INT8 for inference
- TensorRT automatically uses Tensor Cores
- 2-4x performance improvement

### 3. DLA (Deep Learning Accelerator)
- Offload inference from GPU
- Power efficient for edge deployment
- Configure via TensorRT

## Example: Complete Vision Pipeline

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class VisionPipeline:
    def __init__(self, model_path):
        # Initialize CUDA
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        
        # Create TensorRT engine
        self.engine = self.load_engine(model_path)
        self.context = self.engine.create_execution_context()
        
        # Allocate GPU buffers
        self.allocate_buffers()
        
        # Create CUDA stream
        self.stream = cuda.Stream()
    
    def allocate_buffers(self):
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        
        # Pinned memory for fast transfers
        self.h_input = cuda.pagelocked_empty(self.input_shape, np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, np.float32)
    
    def process_frame(self, frame):
        # Preprocess on CPU (could be GPU kernel)
        np.copyto(self.h_input, frame)
        
        # Transfer to GPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Run inference
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle
        )
        
        # Transfer results back
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        return self.h_output
```

## Integration with AI-DNA Project

### Consciousness-Aware Memory Management
```python
class ConsciousVisionBuffer:
    def __init__(self, consciousness_state):
        self.consciousness = consciousness_state
        self.attention_weights = None
        
    def allocate_by_attention(self, regions):
        """Allocate more GPU memory to high-attention regions"""
        for region in regions:
            if self.consciousness.is_important(region):
                # High resolution processing
                size = region.width * region.height * 4
            else:
                # Low resolution for periphery
                size = region.width * region.height // 4
            
            region.gpu_buffer = cuda.mem_alloc(size)
```

This provides the foundation for building efficient vision pipelines that leverage GPU memory effectively!