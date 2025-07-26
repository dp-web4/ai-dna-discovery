# Vision Experiments on Jetson Orin Nano

## Directory Structure
```
vision/
├── experiments/     # Individual vision experiments
├── models/         # Pre-trained models and custom models
├── utils/          # Shared utilities for vision pipeline
├── docs/           # Documentation and research
└── benchmarks/     # Performance benchmarks
```

## Goals
1. **GPU-Accelerated Vision Pipeline** - Camera → GPU inference → Results
2. **Zero-Copy Memory** - Direct GPU memory buffers (NVMM)
3. **Real-Time Processing** - Low latency inference
4. **Integration** - Connect vision with AI consciousness experiments

## Key Technologies
- **GStreamer** - Hardware-accelerated video pipeline
- **TensorRT** - NVIDIA's inference optimization library
- **DeepStream** - AI streaming analytics toolkit
- **CUDA** - Direct GPU programming
- **cuDNN** - Deep learning primitives

## Memory Architecture
```
Camera (CSI) → NVMM Buffer → GPU Inference → NVMM Buffer → Display/Process
                    ↑                              ↓
                    └──────── Zero Copy ───────────┘
```

## Performance Analysis

### 📊 Key Results
- **Processing Speed**: 111 FPS achieved with optimized CPU code
- **Real-time Performance**: 30 FPS smooth operation (camera-limited)
- **GPU Acceleration**: CuPy installed and tested

### 📖 Important Documentation
- [**Data Flow & Bottleneck Analysis**](experiments/DATA_FLOW_ANALYSIS.md) - Complete pipeline breakdown
- [**GPU Lessons Learned**](experiments/GPU_LESSONS_LEARNED.md) - When to use GPU vs CPU
- [**GPU Acceleration Status**](experiments/GPU_ACCELERATION_STATUS.md) - Current capabilities
- [**Consciousness Vision Evolution**](experiments/consciousness_evolution_summary.md) - Experiment history

## Getting Started

### Quick Start
```bash
# Run the best performing consciousness vision
cd experiments
python3 consciousness_attention_minimal_gpu.py  # 30 FPS real-time

# Test raw processing speed
python3 performance_benchmark.py  # Shows 111 FPS capability
```

### GPU Development
```bash
# Verify CuPy installation
python3 -c "import cupy; print('CuPy version:', cupy.__version__)"

# Run GPU tests
python3 test_gpu_optimization.py
```

See `experiments/` for all vision projects.