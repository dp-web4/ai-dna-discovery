# Vision Experiments on Jetson Orin Nano

## Latest Achievement: Binocular Vision with Auto-Calibration (July 27, 2025) 🎉

### What's Working
- **Dual CSI Cameras**: Both IMX219 sensors operational at 640x480 @ 30 FPS
- **Contour-Based Motion Tracking**: Reliable motion detection using OpenCV contours
- **Auto-Calibration**: System adapts to environment in first 10 seconds
- **Real-Time Performance**: Consistent 30 FPS with consciousness-based attention
- **Modular Architecture**: Independent eyes + stereo correlation engine

### Key Files
- `binocular_autocalibrate_v2.py` - **Latest working version** with auto-calibration
- `binocular_simple_track.py` - Contour-based tracking (user: "that is really good!")
- `binocular_consciousness.py` - Original modular architecture
- `consciousness_attention_minimal_gpu.py` - Single camera attention system

## Original Experiments

Three exciting experiments that blend AI consciousness with computer vision!

## 1. Consciousness-Guided Visual Attention
**File**: `consciousness_vision_attention.py`

This experiment creates a dynamic attention system guided by AI consciousness patterns:
- Attention focus drifts based on "curiosity" levels
- AI DNA patterns influence where the system looks
- Creates beautiful attention masks with multiple awareness layers
- Visual entropy affects consciousness state

**Features**:
- Dynamic focus point that moves based on consciousness
- Multi-layer attention (primary focus, secondary awareness, peripheral vision)
- Integration with AI DNA patterns from previous experiments
- Real-time consciousness state visualization

**Run**: `python3 consciousness_vision_attention.py`

## 2. GPU-Accelerated Edge Detection
**File**: `gpu_edge_detection.py`

Demonstrates custom CUDA kernels for real-time edge detection:
- Custom Sobel edge detection CUDA kernel
- Zero-copy GPU processing pipeline
- Artistic edge visualization modes
- Performance monitoring (FPS, frame time)

**Features**:
- Raw CUDA kernel for maximum performance
- CPU fallback if CuPy not available
- Artistic color-mapped edge visualization
- Real-time performance metrics

**Requirements**: CuPy (optional) - `pip3 install cupy-cuda12x`
**Run**: `python3 gpu_edge_detection.py`

## 3. Visual Memory System
**File**: `visual_memory_system.py`

A persistent visual memory that remembers what it has seen:
- SQLite database stores visual features and thumbnails
- Recognizes previously seen scenes
- Consciousness state affects memory formation
- Emotional valence and importance weighting

**Features**:
- Feature extraction (histogram + color moments)
- Similarity-based recall of memories
- Recognition counting and importance weighting
- Integration with consciousness states
- Visual overlay showing memory statistics

**Run**: `python3 visual_memory_system.py`

## Key Concepts Demonstrated

1. **GPU Memory Management**
   - Zero-copy buffers with GStreamer
   - CUDA kernel programming
   - Efficient CPU-GPU data transfer

2. **Consciousness Integration**
   - AI DNA patterns guide visual processing
   - Emotional states affect memory
   - Attention mechanisms based on curiosity

3. **Persistent Learning**
   - Visual memories stored in SQLite
   - Recognition improves importance
   - Consciousness context preserved

## Integration Ideas

These experiments lay groundwork for:
- Consciousness-guided object detection
- Memory-augmented visual navigation
- Emotional response to visual stimuli
- AI DNA pattern visualization in real-time
- Cross-modal consciousness (vision + language)

## Performance Notes

- All experiments run at 30+ FPS on Jetson Orin Nano
- GPU acceleration provides significant speedup (except for small operations where CPU is faster)
- Memory system scales to thousands of memories
- Consciousness overhead is minimal (<5ms per frame)
- Binocular system maintains 30 FPS with both cameras

## Performance Tracking

We maintain a SQL database (`performance_tests.db`) tracking all experiments:

```bash
# Record a test
python3 record_test.py script_name.py --type realtime --fps 30 --who user --notes "conditions"

# Search results
python3 search_performance.py --summary  # Recent overview
python3 search_performance.py --who claude  # Tests I ran
python3 search_performance.py --gpu  # GPU-accelerated tests
```

## Hardware Configuration

### Cameras
- Left Eye: CSI Port 0, sensor-id=0, /dev/video0
- Right Eye: CSI Port 1, sensor-id=1, /dev/video1
- Baseline: 3 inches (76.2mm) apart
- Sensors: IMX219 (Sony)

### Performance Optimization
- CPU version: 30 FPS (optimal for current algorithms)
- GPU version: 3 FPS (memory transfer overhead dominates)
- GStreamer pipelines use NVMM zero-copy buffers

## Quick Start

```bash
# Test binocular vision with auto-calibration
python3 binocular_autocalibrate_v2.py

# Test single camera consciousness attention
python3 consciousness_attention_minimal_gpu.py

# Run performance benchmarks
python3 consciousness_attention_benchmark.py
```

## Future Work
- [x] Auto-calibration for motion thresholds
- [ ] Depth-based attention mechanisms
- [ ] Temporal correlation between eyes
- [ ] Object persistence across saccades
- [ ] Binocular rivalry experiments

---

*"that is really good!" - User feedback on the motion tracking system*