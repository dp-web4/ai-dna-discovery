# IMU Visualizer Performance Guide

## Performance Versions

### 1. **imu_visualizer_lite.py** (Recommended for performance)
- **Target FPS**: 60+
- **Features**: Essential plots only
- **Optimizations**:
  - Pre-created graphics objects
  - Matplotlib blitting for fast updates
  - Minimal plot count (4 plots vs 6)
  - Efficient data structures
  - 16ms update interval

### 2. **imu_visualizer_v2.py** (Balanced)
- **Target FPS**: 50
- **Features**: Full feature set with gyro time series
- **Optimizations**:
  - Thread-safe data access
  - Deque for history management
  - 20ms update interval

### 3. **imu_visualizer.py** (Original)
- **Target FPS**: 20
- **Features**: Complete visualization
- **Note**: Lower performance but most features

## Orientation Configuration

### Problem
IMU and cameras may not be aligned when mounted together. The IMU axes might not match camera axes.

### Solution
Use `imu_orientation_mapper.py` to configure the mapping:

```bash
# Interactive setup
python3 imu_orientation_mapper.py

# Test current configuration
python3 imu_orientation_mapper.py test
```

### Common Configurations

1. **Default** (IMU and camera aligned):
   - X → X, Y → Y, Z → Z

2. **Upside Down** (IMU mounted inverted):
   - X → X, Y → -Y, Z → -Z
   - Roll is flipped

3. **Rotated 90° on Z** (IMU rotated in mounting plane):
   - X → Y, Y → -X, Z → Z
   - 90° yaw offset

4. **Side Mount** (IMU on its side):
   - X → Z, Y → Y, Z → -X
   - 90° pitch offset

### Integration with Vision System

The stabilized vision system automatically loads the orientation configuration:

```python
# In imu_stabilized_vision.py
config = OrientationConfig.load()  # Loads imu_orientation_config.json
mapper = IMUOrientationMapper(config)
```

## Performance Tips

### For Maximum Performance
1. Use `imu_visualizer_lite.py`
2. Close other applications
3. Reduce serial baud rate if needed (though 115200 should be fine)
4. Disable unneeded plots in the code

### Debugging Slow Frame Rate
1. Check CPU usage with `htop`
2. Monitor IMU data rate in the status display
3. Try reducing update interval (but may cause instability)
4. Check for other serial port users

### Hardware Mounting Tips
1. Mount IMU rigidly (no vibration)
2. Keep IMU away from magnetic interference
3. Align IMU axes with camera as closely as possible
4. Document your mounting orientation for future reference

## Quick Test Commands

```bash
# Lightweight visualizer (best performance)
./imu_visualizer_lite.py

# Or with sudo if needed
sudo python3 imu_visualizer_lite.py

# Configure orientation
python3 imu_orientation_mapper.py

# Test with stabilized vision
cd ../vision/experiments
./launch_stabilized_vision.sh
```