# Coherence Engine

The Coherence Engine implements reality field generation through weighted sensor fusion, where memory and cognition are temporal sensors alongside spatial ones.

## Core Principle

```
Reality Field = f(Spatial Sensors, Temporal Sensors, Context, Trust)
```

## Architecture

```
coherence-engine/
├── core/               # Platform-agnostic engine
│   ├── engine.py      # Main coherence engine
│   ├── context.py     # Context state management
│   ├── trust.py       # Trust evolution system
│   └── attention.py   # Attention policy
├── plugins/           # Platform-specific implementations
│   ├── common/        # Shared plugin base
│   ├── jetson/        # Jetson-specific (CSI cameras, IMU)
│   └── legion/        # Legion-specific (webcam, GPU, audio)
├── memory/            # Experience storage
├── docs/              # Documentation
└── tests/             # Test suite

```

## Platform Support

### Jetson (Edge Device)
- Dual CSI cameras for vision (1920x1080 @ 30 FPS)
- Yahboom CMP10A IMU for motion (6-axis + magnetometer)
- Real-time video dashboard with coherence visualization
- GPIO access for hardware control
- Edge AI inference

### Legion (RTX 4090)
- GPU sensors via nvidia-smi (utilization, temperature)
- Audio input via PyAudio (real microphone monitoring)
- Webcam/USB cameras (OpenCV support)
- Auto-detection of available sensors
- Fallback to simulated data when hardware unavailable

### Common Features
- Memory as temporal sensor
- Cognition sensor (LLM integration)
- Network bridge for distributed consciousness
- Context-aware trust evolution
- Reality field generation

## Quick Start

### On Jetson
```bash
cd ai-dna-discovery/coherence-engine
python3 plugins/jetson/run_jetson.py
```

### On Legion
```bash
# Install dependencies first
sudo apt-get install portaudio19-dev libasound2-dev
pip3 install pyaudio numpy

# Run coherence engine
cd ai-dna-discovery/coherence-engine
python3 run_legion.py
```

## Core Concepts

### Sensors
Each sensor provides a `read()` method returning normalized [0,1] values representing their contribution to the reality field.

### Trust Evolution
Sensors earn trust through accurate predictions and lose trust through conflicts or errors. Trust is tracked per-context.

### Context States
- **STABLE**: High spatial trust, low temporal activity
- **MOVING**: Balanced spatial/temporal, moderate memory
- **UNSTABLE**: Low peripheral trust, high attention
- **NOVEL**: High memory search, high cognition

### Attention Policy
Triggers context transitions based on:
- Prediction errors
- Sensor conflicts
- Confidence drops
- Resource constraints

## Plugin Development

To add a new sensor:

1. Create plugin in `plugins/[platform]/[sensor_name].py`
2. Implement the sensor interface:
```python
class MySensor:
    id = "my_sensor"
    
    def read(self, *, tick: int) -> float:
        # Return normalized [0,1] value
        return sensor_value
```

3. Register in platform configuration
4. The engine will automatically integrate it

## Cross-Platform Testing

The same coherence engine runs on all platforms with different sensor configurations:

- **Jetson**: Physical sensors (camera, IMU)
- **Legion**: Compute sensors (GPU, CPU)
- **Both**: Memory, cognition, network

This allows testing the same consciousness principles across different hardware capabilities.