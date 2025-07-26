# Performance Test Tracking System

## Overview
We now have a SQL database to accurately track all performance tests and their results.

## Database Schema

### test_runs table
- **id**: Unique test identifier
- **timestamp**: When the test was run
- **filename**: Script that was tested
- **test_type**: realtime, benchmark, or stress
- **fps_avg/min/max**: Frame rate statistics
- **processing_time_ms**: Per-frame processing time
- **gpu_used**: Whether GPU acceleration was used
- **gpu_library**: Which GPU library (cupy, vpi, tensorrt)
- **resolution**: Test resolution
- **notes**: Additional context
- **who_ran**: user, claude, or automated

### test_details table
- Additional metrics for each test run
- Linked to test_runs via foreign key

## Usage

### Record a new test
```bash
python3 record_test.py consciousness_attention_new.py --fps 45.2 --gpu --gpu-lib cupy --notes "Testing new optimizations"
```

### Search tests
```bash
# Show summary
python3 search_performance.py --summary

# Find GPU tests
python3 search_performance.py --gpu

# Find tests by Claude
python3 search_performance.py --who claude

# Find high-performance tests
python3 search_performance.py --min-fps 50

# Show details for specific test
python3 search_performance.py --details 1
```

## Verified Test Results

### 1. Performance Benchmark (ID: 1)
- **File**: performance_benchmark.py
- **Who ran**: Claude (automated test)
- **FPS**: 118.9 (theoretical max)
- **Processing**: 8.41ms total
- **GPU**: No
- **Notes**: CPU benchmark showing theoretical capabilities

### 2. GPU Consciousness Vision (ID: 2)
- **File**: consciousness_attention_gpu.py
- **Who ran**: User
- **FPS**: ~3 (very slow)
- **GPU**: Yes (CuPy)
- **Notes**: Memory transfer overhead dominated

### 3. Minimal GPU Version (ID: 3)
- **File**: consciousness_attention_minimal_gpu.py
- **Who ran**: User
- **FPS**: 30 (camera-limited)
- **GPU**: No
- **Notes**: Optimized CPU version, smooth real-time

### 4. Clean Consciousness (ID: 4)
- **File**: consciousness_attention_clean.py
- **Who ran**: User
- **FPS**: 20-25
- **GPU**: No
- **Notes**: Best biological model with good motion response

## Important Clarifications

1. **The 111 FPS claim** came from Claude's benchmark test showing theoretical CPU processing capability, not from actual camera tests

2. **Real-world performance**:
   - Camera tests are limited to 30 FPS by GStreamer pipeline
   - Actual achieved FPS: 20-30 depending on algorithm
   - GPU version was slower (3 FPS) due to transfer overhead

3. **Bottlenecks identified**:
   - GStreamer pipeline configuration (30 FPS cap)
   - CPU-GPU memory transfers (14ms overhead)
   - OpenCV drawing operations (CPU-only)

## Best Practices

1. **Always record test conditions**:
   - Who ran it (user vs automated)
   - What type of test (realtime vs benchmark)
   - Full context in notes

2. **Distinguish between**:
   - Theoretical benchmarks (no camera)
   - Real-time tests (with camera)
   - Stress tests (extreme conditions)

3. **Include all relevant metrics**:
   - Not just FPS but also min/max
   - Processing time per frame
   - Which libraries were used

This tracking system ensures we maintain accurate records and don't lose track of what actually happened in our experiments.