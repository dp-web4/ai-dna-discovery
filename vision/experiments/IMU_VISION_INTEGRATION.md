# IMU-Vision Integration

## Overview
Integration of Yahboom CMP10A IMU with binocular vision system for stabilization and head tracking.

## Components

### 1. IMU-Stabilized Vision (`imu_stabilized_vision.py`)
Real-time video stabilization using IMU orientation data:
- Compensates for camera/head movement
- Smooths rotation using rolling average
- Toggle stabilization on/off with 's' key
- Reset reference orientation with 'r' key

### 2. Head Tracking System (`imu_head_tracker.py`)
Advanced head tracking with gaze prediction:
- Converts head orientation to gaze direction
- Detects saccades from angular velocity
- Attention-based visual processing
- Motion prediction for smoother tracking

### 3. Launch Script (`launch_stabilized_vision.sh`)
Safe launcher that handles permission issues:
- Checks serial port access
- Offers sudo option if needed
- Falls back to vision-only mode

## Technical Details

### IMU Data Processing
- **Orientation**: Roll, Pitch, Yaw in degrees
- **Angular Velocity**: 3-axis gyroscope data (deg/s)
- **Linear Acceleration**: 3-axis accelerometer (g)
- **Update Rate**: ~100Hz from IMU, processed at camera FPS

### Stabilization Algorithm
1. Calculate orientation delta from reference
2. Apply rolling average filter (10 samples)
3. Generate 2D rotation matrix for yaw
4. Add translation compensation for pitch/roll
5. Apply affine transformation to frames

### Head Tracking Features
- **Gaze Mapping**: Head angles → normalized coordinates
- **Saccade Detection**: Angular velocity > 50°/s
- **Motion Prediction**: Linear extrapolation
- **Attention Map**: Gaussian-weighted focus area

## Usage

### Quick Start
```bash
# With permission handling
./launch_stabilized_vision.sh

# Direct execution (requires port access)
python3 imu_stabilized_vision.py

# Head tracking demo
python3 imu_head_tracker.py
```

### Controls
- **s**: Toggle stabilization ON/OFF
- **r**: Reset reference orientation
- **q/ESC**: Quit application

### Permission Setup
If you get permission errors:
```bash
# Option 1: Add to dialout group (permanent)
sudo usermod -a -G dialout $USER
# Then logout and login

# Option 2: Temporary fix
sudo chmod 666 /dev/ttyUSB0

# Option 3: Run with sudo
sudo python3 imu_stabilized_vision.py
```

## Integration Points

### With Binocular Vision
- Stabilizes both camera feeds independently
- Maintains stereo alignment during movement
- Reduces motion blur in tracking

### With Consciousness System
- Head orientation influences attention
- Saccade detection triggers focus shifts
- Gaze prediction for anticipatory processing

### Future Enhancements
1. **3D Stabilization**: Full rotation compensation
2. **Kalman Filtering**: Better motion estimation
3. **Eye-Head Coordination**: Model VOR (vestibulo-ocular reflex)
4. **Learning**: Adapt to user's head movement patterns

## Performance Considerations
- IMU adds ~2ms latency
- Stabilization computation: ~5ms per frame
- Total overhead: <10ms (maintains 30 FPS)
- GPU acceleration possible via CUDA

## Troubleshooting

### IMU Not Detected
- Check USB connection: `ls -la /dev/ttyUSB*`
- Verify baud rate: Should be 115200 (configured from 9600)
- Test with IMU tools: `../../imu/imu_config_tool.py`

### Stabilization Jittery
- Increase smoothing history size
- Check IMU mounting (should be rigid)
- Calibrate IMU if needed

### Performance Issues
- Reduce resolution if needed
- Disable debug overlays
- Use GPU acceleration for transforms