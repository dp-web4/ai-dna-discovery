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

## Getting Started
See `experiments/` for specific vision projects.