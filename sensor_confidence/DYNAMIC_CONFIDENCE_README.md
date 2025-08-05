# Dynamic Sensor Confidence Framework

## Core Philosophy: Trust is Never Absolute

In the real world, sensor reliability isn't binary. A magnetometer doesn't suddenly stop working at 90° tilt - its confidence gradually degrades as orientation changes. This framework implements that reality.

## Key Concepts

### 1. Continuous Confidence Scaling
- **Horizontal (0° tilt)**: 100% magnetometer confidence
- **45° tilt**: 50% confidence 
- **Vertical (90° tilt)**: 15% confidence
- Uses exponential decay: `confidence = e^(-k * tilt_angle)`

### 2. Contextual Modifiers
Every reading includes context that affects confidence:
- **Near metal**: Reduces magnetometer confidence by 50%
- **High acceleration**: Indicates non-gravity forces, reduces confidence
- **Rapid rotation**: Sensor may be near saturation limits
- **Instability**: Recent erratic movement reduces trust

### 3. Temporal Awareness
- Tracks stability over time
- Penalizes rapid changes
- Historical reliability influences current confidence

## Implementation

### Core Components

1. **`dynamic_magnetometer_confidence.py`**: Standalone dynamic confidence calculator
   - Real-time tilt calculation
   - Stability tracking
   - Contextual adjustments

2. **`confidence_framework.py`**: Updated with dynamic magnetometer confidence
   - Replaces static vertical mount detection
   - Integrates tilt-based confidence

3. **`imu_vertical_confidence.py`**: Enhanced calibration tool
   - Now detects actual mounting orientation
   - Provides axis remapping suggestions

### Usage Example

```python
from sensor_confidence.dynamic_magnetometer_confidence import DynamicMagnetometerConfidence

# Create confidence tracker
mag_conf = DynamicMagnetometerConfidence()

# Update with current accelerometer data
result = mag_conf.update_confidence(
    accel_x=0.5,  # m/s²
    accel_y=2.0,  # m/s²
    accel_z=9.5,  # m/s²
    context={"near_metal": False}
)

print(f"Tilt: {result.tilt_angle:.0f}°")
print(f"Magnetometer confidence: {result.magnetometer_confidence:.0%}")
print(f"Stability: {result.stability:.0%}")
```

## Real-World Scenarios

### Drone/Robot Navigation
- Level flight: High magnetometer confidence for heading
- Banking turn: Reduced confidence, rely more on gyro
- Vertical climb: Minimal magnetometer trust

### Handheld Device
- Normal use: Good confidence
- Device in pocket (random orientations): Variable confidence
- Near keys/metal: Context reduces confidence further

### Vehicle Mount
- Dashboard (mostly level): High confidence
- Off-road tilting: Dynamic confidence adjustment
- Near engine: Context modifier for magnetic interference

## Confidence Curve

The magnetometer confidence follows this curve:

```
100% |*
 80% | **
 60% |   ***
 40% |      ****
 20% |          *******
  0% +------------------
     0°  30°  60°  90°
     Tilt from horizontal
```

## Integration with Sensor Fusion

Each sensor provides a tuple: `(value, confidence, context)`

This allows fusion algorithms to:
- Weight sensors by current confidence
- Detect when to switch primary sensors
- Provide uncertainty estimates

## Web4 Principles Embodied

1. **Data Quality is Contextual**: Same sensor, different confidence based on situation
2. **Trust is Temporal**: What was reliable a second ago might not be now
3. **Transparency**: System knows and reports when it doesn't know
4. **Graceful Degradation**: Reduced confidence, not binary failure

## Testing

Run the demo to see dynamic confidence in action:

```bash
python3 sensor_confidence/dynamic_magnetometer_confidence.py
```

This will simulate various tilts and show how confidence adapts.

## Future Enhancements

1. **Multi-axis confidence**: Different confidence for different measurement axes
2. **Environmental learning**: Adapt to specific deployment conditions
3. **Sensor fusion feedback**: Use other sensors to validate/adjust confidence
4. **Predictive confidence**: Anticipate confidence changes based on motion patterns