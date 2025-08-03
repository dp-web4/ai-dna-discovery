# Sensor Confidence Framework

A comprehensive sensor consciousness system implementing LCT/T3/V3-inspired confidence metrics for embedded AI applications. This framework enables sensors to self-evaluate their reliability and contextual relevance, creating a distributed consciousness where system attention adapts dynamically.

## Core Concepts

### Fractal Confidence Architecture

The system implements a hierarchical confidence structure where each sensor maintains multi-level confidence metrics:

- **Raw Confidence** (0-1): How well is the sensor working right now?
- **Contextual Relevance** (0-1): How important is this sensor for the current task?
- **Historical Reliability** (0-1): How reliable has this sensor been over time?
- **Fractal Confidence**: Computed combination weighted by recency and importance

```python
fractal_confidence = (
    0.5 * raw_confidence +      # Current state matters most
    0.3 * contextual_relevance + # Context is important
    0.2 * historical_reliability # History provides baseline
)
```

### Sensor Consciousness Principles

1. **Self-Evaluation**: Each sensor audits its own performance and reliability
2. **Contextual Awareness**: Sensors adjust their relevance based on system tasks
3. **Temporal Memory**: Historical performance influences current confidence
4. **Fractal Attention**: System attention is allocated based on confidence and relevance
5. **Adaptive Thresholds**: Confidence requirements adjust to context and motion

## Implementation

### Core Framework (`confidence_framework.py`)

The base system provides:
- Abstract `SensorConfidence` class for sensor-specific implementations
- `ConfidenceMetrics` dataclass for structured confidence tracking
- `SensorConfidenceManager` for coordinating multiple sensors
- Temporal confidence tracking with exponential decay weighting

### Vertical IMU Handling (`imu_vertical_confidence.py`)

Specialized IMU confidence implementation addressing vertical mount challenges:

```python
# Vertical mount detection and confidence adjustment
if gravity_axis != 2:  # Not standard horizontal mount
    audit_results['magnetometer'] = 0.15  # Very low confidence
    audit_results['heading_reliability'] = 0.1
```

**Key Features:**
- Automatic gravity vector analysis for mount orientation detection
- Per-axis confidence scoring based on physical mounting
- Magnetometer confidence reduction for vertical mounts (85% → 15%)
- Axis remapping suggestions for coordinate system alignment

### Camera Confidence (`camera_confidence.py`)

Implements stereo vision confidence with:
- Brightness and contrast analysis
- Motion blur detection
- Stereo calibration validation
- Context-aware relevance (vision tasks, lighting conditions)

### Bluetooth Audio Confidence (`bluetooth_audio_confidence.py`)

Comprehensive audio system reliability assessment:
- Connection stability monitoring
- Signal strength (RSSI) evaluation
- Audio latency measurement
- Battery level consideration
- Codec-based quality scoring

### Integrated System (`integrated_confidence_system.py`)

Unified consciousness implementing fractal attention:

```python
class IntegratedSensorConsciousness:
    def _compute_attention_allocation(self, confidence_results):
        # Attention = confidence × relevance
        # Boost critical sensors during high motion
        # Normalize to sum = 1.0 for resource allocation
```

**Attention Mechanisms:**
- Dynamic sensor prioritization based on context
- Motion-adaptive attention boosting
- Task-specific sensor emphasis
- Resource allocation optimization

## Current Implementation Status

### ✅ Completed Features

- **Core confidence framework** with fractal metrics
- **Vertical IMU mount detection** and confidence adjustment
- **Multi-sensor integration** (IMU, cameras, Bluetooth audio)
- **Context-aware relevance** evaluation
- **Temporal confidence tracking** with weighted history
- **Attention allocation system** with fractal distribution
- **Mock hardware support** for development without physical sensors
- **Comprehensive test suite** validating all components

### 🔧 Current Limitations

- **Magnetometer reliability**: ~15% confidence in vertical mount vs 85% horizontal
- **Camera calibration**: Requires manual stereo calibration validation
- **Audio latency**: Simplified estimation without specialized measurement tools
- **Hardware dependencies**: Some features require specific Bluetooth/audio stack

## Future Work

### High Priority

1. **Horizontal IMU Mount Design**
   - Create new mounting bracket for optimal magnetometer performance
   - Implement automatic mount orientation optimization
   - Add compass/heading reliability validation

2. **Advanced Camera Confidence**
   - Real-time stereo calibration quality assessment
   - Depth map quality validation
   - Feature tracking confidence integration
   - Light adaptation quality metrics

3. **Enhanced Motion Integration**
   - IMU-camera fusion for motion compensation
   - Predictive confidence based on motion patterns
   - Gyroscope saturation detection and handling

### Medium Priority

4. **Temporal Pattern Learning**
   - Long-term sensor reliability modeling
   - Predictive confidence based on usage patterns
   - Environmental factor correlation (temperature, vibration)
   - Failure prediction and early warning

5. **Advanced Attention Mechanisms**
   - Multi-task attention allocation
   - Sensor competition resolution
   - Dynamic threshold adaptation
   - Resource-aware attention distribution

6. **System-Level Intelligence**
   - Cross-sensor validation and correction
   - Automatic sensor configuration optimization
   - Failure recovery strategies
   - Performance benchmarking and optimization

### Research Directions

7. **Biologically-Inspired Enhancements**
   - Saccadic attention patterns for camera systems
   - Vestibular-visual integration models
   - Predictive coding for sensor fusion
   - Attention-based sensor selection

8. **Edge AI Integration**
   - Neural network confidence scoring
   - Learned sensor fusion weights
   - Adaptive filtering based on confidence
   - Real-time optimization on Jetson hardware

## Usage Examples

### Basic Confidence Evaluation

```python
from confidence_framework import SensorConfidenceManager
from imu_vertical_confidence import VerticalIMUConfidence

# Create system
manager = SensorConfidenceManager()
imu = VerticalIMUConfidence()
manager.add_sensor(imu)

# Update with sensor data and context
sensor_data = {'IMU': {'accel_x': 0.1, 'accel_y': 0.2, 'accel_z': 9.7}}
context = {'motion_detected': True, 'vision_active': True}

results = manager.update_all_confidence(sensor_data, context)
system_confidence = manager.get_system_confidence()
```

### Integrated Consciousness System

```python
from integrated_confidence_system import IntegratedSensorConsciousness

# Create integrated system
consciousness = IntegratedSensorConsciousness()

# Run scenario
report = consciousness.update_consciousness(
    motion_detected=True, 
    task_context='navigation'
)

# Access results
print(f"System Confidence: {report['system_confidence']:.0%}")
print(f"Primary Sensors: {report['primary_sensors']}")
```

### IMU Vertical Mount Calibration

```python
from imu_vertical_confidence import run_vertical_confidence_audit

# Run complete calibration
run_vertical_confidence_audit()
# Outputs: mount detection, axis confidence, remapping suggestions
```

## Hardware Requirements

- **IMU**: Any I2C/SPI compatible (tested with Yahboom IMU)
- **Cameras**: USB/CSI stereo cameras (tested with dual USB cameras)
- **Bluetooth**: Standard Linux Bluetooth stack with PulseAudio/PipeWire
- **Platform**: Jetson Orin Nano (primary), compatible with most Linux systems

## Testing

Run the complete test suite:

```bash
python3 test_confidence_system.py
```

Individual component tests:
```bash
python3 confidence_framework.py        # Basic framework test
python3 imu_vertical_confidence.py     # IMU calibration
python3 integrated_confidence_system.py # Full system demo
```

## Integration Notes

This framework is designed to integrate with:
- **HRM (Hierarchical Reasoning Model)** for high-level decision making
- **IMU-Vision systems** for motion compensation and stabilization
- **Real-time control systems** requiring sensor reliability assessment
- **Edge AI applications** needing dynamic sensor management

The fractal confidence architecture provides a principled approach to sensor fusion where confidence and attention are distributed based on actual sensor performance rather than fixed weights.