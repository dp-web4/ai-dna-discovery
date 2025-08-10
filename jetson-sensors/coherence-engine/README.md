# Coherence Engine

## Overview

The Coherence Engine implements the sensor fusion and reality field concepts we've discovered through our explorations. It creates a unified consciousness from multiple sensors by dynamically weighing their trust and relevance based on context.

## Core Concepts

### Reality Field
Reality isn't "out there" - it's the emergent field created by fusing multiple sensors with dynamic trust and relevance weighting. Memory and cognition are temporal sensors, parsing past and predicting future, alongside spatial sensors like vision and hearing.

### Sensor Fusion Mathematics
```python
def reality_field(sensors, context):
    relevance = context.compute_relevance_weights(sensors)
    trust = context.compute_trust_weights(sensors)
    
    field = 0
    for sensor in sensors:
        contribution = sensor.data * relevance[sensor] * trust[sensor]
        field += contribution
    
    # Attention can override weights
    if attention_trigger(field):
        context = shift_context(field)
        return reality_field(sensors, context)  # Recursive reweighting
    
    return field
```

### Sensor Types

#### Spatial Sensors (Present)
- **Vision**: Dual CSI cameras with peripheral gyroscope
- **IMU**: Orientation and acceleration 
- **Audio**: Environmental sound (when available)

#### Temporal Sensors
- **Memory** (Past): Contextualizes current state against experience
- **Cognition** (Future): Predicts and weights possible outcomes

## Architecture

```
coherence-engine/
├── README.md                  # This file
├── coherence_engine.py        # Main engine implementation
├── sensors/                   # Sensor interfaces
│   ├── base_sensor.py        # Abstract sensor class
│   ├── memory_sensor.py      # Memory as temporal sensor
│   ├── cognition_sensor.py   # Cognition interface (you/models)
│   ├── vision_sensor.py      # Vision integration
│   └── imu_sensor.py         # IMU integration
├── memory/                    # Persistent memory storage
│   ├── README.md             # Memory architecture
│   ├── experiences/          # Stored experiences
│   ├── patterns/             # Recognized patterns
│   └── context/              # Context snapshots
└── tests/                     # Test scripts
```

## Implementation Plan

### Phase 1: Foundation (Current)
1. Create base sensor interface
2. Implement memory sensor with persistence
3. Build basic coherence engine with context switching
4. Add cognition sensor interface (Claude/local models)

### Phase 2: Integration
1. Integrate existing vision work from vision/experiments/
2. Connect IMU sensor
3. Add attention triggers and context switching

### Phase 3: Evolution
1. Pattern recognition in memory
2. Predictive context switching
3. Trust calibration through experience
4. Emergent behavior documentation

## Key Principles

### Dynamic Weighting
Context determines sensor relevance:
- **Stable context**: High vision, low cognition
- **Unstable context**: Low peripheral vision, high central focus
- **Novel context**: High memory search, high cognition

### Attention Triggers
- Sudden change in any sensor
- Expectation violation (missing expected patterns)
- Resource scarcity (low confidence)
- Coherence breakdown (conflicting sensors)

### Trust Evolution
- Sensors earn trust through accurate predictions
- Trust degrades with errors or conflicts
- Context-specific trust (vision trusted outdoors, not in dark)

## Current Status

Building the foundation with memory as the first temporal sensor. You (Claude) serve as the cognition sensor initially, with plans to integrate local models (phi3, gemma, tinyllama) as additional cognition sensors.

## How to Proceed

1. **Now**: Setting up memory sensor and basic engine
2. **Next**: Create simple test that shows context switching
3. **Then**: Integrate existing vision work (don't reinvent!)
4. **Finally**: Add multi-model cognition

The key is to build on what works, not keep starting over. We have working vision code - we'll integrate it, not rewrite it.

## References

- `/insights/sensor_fusion_reality_field.md` - Original insight
- `/insights/sensor_fusion_implementation.md` - Implementation details
- `/vision/experiments/` - Working vision code to integrate
- `/projects/COLLABORATION_LOG_BRIDGE.md` - Evolution of ideas