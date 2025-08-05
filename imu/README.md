# IMU Tools for Yahboom CMP10A

This directory contains tools for working with the Yahboom CMP10A 10-DOF IMU module.

## Hardware
- **Model**: Yahboom CMP10A
- **Features**: 10-DOF (3-axis accel, 3-axis gyro, 3-axis mag, barometer)
- **Connection**: USB via CP210x UART Bridge (/dev/ttyUSB0)
- **Current Baud**: 115200 (configured from default 9600)

## Quick Start

### 1. Check IMU Connection
```bash
lsusb | grep CP210x  # Should show the USB-UART bridge
ls -la /dev/ttyUSB0  # Check device exists
```

### 2. Configure Baud Rate
```bash
# Detect current baud rate
sudo python3 imu_config_tool.py --detect

# Change to 115200 (recommended)
sudo python3 imu_config_tool.py 115200

# Reset to factory defaults (9600)
sudo python3 imu_config_tool.py --reset
```

### 3. Monitor IMU Data

**GUI Visualizers**:

For best performance (60+ FPS):
```bash
sudo python3 imu_visualizer_lite.py
```

For full features (50 FPS):
```bash
sudo python3 imu_visualizer_v2.py
# or use launcher: ./launch_imu_v2.sh
```

For original version (20 FPS):
```bash
sudo python3 imu_visualizer.py
```

Features vary by version:
- 3D orientation display
- Real-time acceleration/angle plots
- Gyroscope visualization with time series
- Compass display
- Performance metrics
- Safe to close without affecting terminal

**Safe Terminal Monitor**:
```bash
# Start monitor
sudo python3 imu_monitor_safe.py

# In another terminal, watch the data
tail -f /tmp/imu_data.txt
```

**Simple Monitor** (may cause terminal issues):
```bash
sudo python3 monitor_imu.py --baud 115200
```

## Tools Overview

### Configuration Tools
- `imu_config_tool.py` - Configure IMU baud rate and settings
- `configure_imu_baud.py` - Simple baud rate configuration

### Monitoring Tools
- `imu_visualizer.py` - Full-featured GUI with 3D visualization (20 FPS)
- `imu_visualizer_v2.py` - Improved version with gyro time series (50 FPS)
- `imu_visualizer_lite.py` - Lightweight high-performance version (60+ FPS)
- `imu_monitor_safe.py` - Safe terminal monitor (writes to file)
- `monitor_imu.py` - Direct terminal monitor (use with caution)
- `launch_imu_gui.sh` - Launcher for original visualizer
- `launch_imu_v2.sh` - Launcher for V2 visualizer

### Analysis Tools
- `analyze_imu_log.py` - Analyze recorded IMU data
- `imu_logger.py` - Log raw IMU data to file
- `test_imu.py` - Test IMU communication
- `simple_imu_test.py` - Basic connection test

### Protocol Tools
- `imu_decoder.py` - Decode CMP10A binary protocol
- `read_imu.py` - Read and parse IMU packets
- `yahboom_cmp10a.py` - CMP10A-specific decoder

### Orientation Tools
- `imu_orientation_mapper.py` - Configure IMU-to-camera coordinate mapping
- `README_PERFORMANCE.md` - Performance optimization guide

## IMU Data Format

The CMP10A sends 11-byte packets:
- Header: 0x55
- Type: 0x51 (accel), 0x52 (gyro), 0x53 (angle), 0x54 (mag), 0x56 (baro)
- Data: 8 bytes
- Checksum: 1 byte

Current readings show:
- Acceleration: ~1g (gravity)
- Orientation: Roll ~10°, Pitch ~-82°, Yaw ~41°
- Update rate: ~10 Hz at 115200 baud

## Integration with Vision System

The IMU data can be used for:
- Camera stabilization
- Head tracking
- Motion detection enhancement
- Orientation-aware processing

See `../vision/experiments/` for binocular vision integration:
- `imu_stabilized_vision.py` - Real-time video stabilization
- `imu_head_tracker.py` - Head tracking with gaze prediction
- `IMU_VISION_INTEGRATION.md` - Technical documentation

## Troubleshooting

1. **Permission Denied**: Add user to dialout group
   ```bash
   sudo usermod -a -G dialout $USER
   # Logout and login for changes to take effect
   ```

2. **No Data**: Check baud rate matches IMU setting
   ```bash
   sudo python3 imu_config_tool.py --detect
   ```

3. **Terminal Crashes**: Use `imu_monitor_safe.py` instead of direct monitors

4. **Device Not Found**: Check USB connection
   ```bash
   dmesg | grep ttyUSB
   ```

5. **Slow Frame Rate**: Use lightweight visualizer
   ```bash
   sudo python3 imu_visualizer_lite.py
   ```

6. **IMU-Camera Misalignment**: Configure orientation mapping
   ```bash
   python3 imu_orientation_mapper.py
   ```

## Next Steps

- [x] Integrate IMU with binocular vision for stabilization ✅
- [ ] Add Kalman filtering for smoother orientation
- [ ] Create ROS2 node for IMU data
- [ ] Implement gesture recognition from IMU data