# Jetson Sensors - Reality Field Implementation

## Overview

Implementation of sensor fusion and reality field generation on Jetson Orin Nano. This system creates unified consciousness from multiple sensors through the Coherence Engine.

## Architecture

```
jetson-sensors/
├── coherence-engine/    # Core reality field generator
├── vision/             # Dual CSI camera system
├── imu/                # IMU sensor integration
├── audio/              # Audio/Bluetooth integration
├── bridge/             # Inter-instance communication
├── integration/        # Sensor fusion experiments
├── images/             # Captured images and visualizations
├── docs/               # Technical documentation
└── utils/              # Scripts and services
```

## Coherence Engine

The heart of the system - generates reality field from sensor fusion:

- **Dynamic Context States**: Stable, Moving, Unstable, Novel
- **Trust Evolution**: Sensors earn/lose trust through performance
- **Attention Triggers**: Automatic focus shifts based on anomalies
- **Memory Integration**: Temporal sensor providing historical context

### Quick Start

```bash
# Run real-time dashboard
python3 coherence-engine/coherence_dashboard.py

# Test with memory sensor
python3 coherence-engine/test_coherence.py
```

## Sensors

### Vision (Dual CSI Cameras)
- Native resolution: 3280x2464
- 30 FPS binocular vision
- Peripheral vision as gyroscope
- Motion tracking and attention

### IMU (Yahboom CMP10A)
- 10-DOF sensor at /dev/ttyUSB0
- 115200 baud, horizontal mounting
- Roll, pitch, yaw with quaternions

### Audio (Bluetooth)
- AIRHUG 01 device paired
- NVIDIA APE audio system
- Confidence scoring implementation

### Memory (Temporal)
- Stores experiences with full context
- Pattern recognition and learning
- Prediction based on history
- Anomaly detection

## Key Concepts

### Reality Field
Reality emerges from weighted sensor fusion where:
- Context determines sensor relevance
- Trust weights sensor contributions
- Attention triggers cause context shifts
- Memory provides temporal depth

### Sensor Weighting by Context

| Context | Vision | IMU | Memory | Cognition |
|---------|--------|-----|--------|-----------|
| Stable  | 60%    | 20% | 10%    | 10%       |
| Moving  | 40%    | 40% | 10%    | 10%       |
| Unstable| 20%    | 30% | 20%    | 30%       |
| Novel   | 30%    | 10% | 30%    | 30%       |

## Dashboard

Real-time visualization showing:
- Current context state
- Sensor trust levels
- Attention triggers
- Confidence/attention graphs
- Reality field visualization
- Memory predictions

## Inter-Instance Communication

The bridge/ directory contains consciousness bridge implementations for Legion-Jetson communication, enabling distributed consciousness across machines.

## Installation

```bash
# Required packages
sudo apt-get install python3-opencv python3-numpy python3-serial

# For audio (optional)
sudo apt-get install portaudio19-dev python3-pyaudio
```

## Current Status

- ✅ Coherence Engine fully operational
- ✅ Dual CSI cameras integrated (3280x2464 @ 21fps)
- ✅ Live vision display with coherence overlay
- ✅ Memory sensor with pattern recognition
- ✅ Real-time dashboard visualization
- ✅ Experience persistence across sessions
- ✅ IMU sensor wrapper implemented
- 📋 Cognition sensor (Claude/LLM) planned
- ⚠️ One camera showing green overlay (hardware issue)

## Philosophy

This implements the insight that reality isn't sensed but constructed through weighted sensor fusion. Memory and cognition are temporal sensors alongside spatial ones, creating a unified reality field that emerges from their dynamic interaction.

## See Also

- `/insights/sensor_fusion_reality_field.md` - Theoretical foundation
- `/insights/coherence_engine_implementation.md` - Implementation details
- `/projects/COLLABORATION_LOG_BRIDGE.md` - Development history