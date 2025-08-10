# Vision Setup - Dual CSI Cameras

## ✅ Working Configuration

### Camera Hardware
- **Dual IMX219 CSI cameras** on Jetson Orin Nano
- **Manual focus rings** on each camera (important!)
- **Resolution**: 1920x1080 @ 30fps (sensor mode 2)
- **Native max**: 3280x2464 @ 21fps (sensor mode 0)

### Key Learnings

1. **Focus Issue Solved**: Right camera blur was due to manual focus ring being at different position than left camera. Adjust both focus rings to match!

2. **Optimal Mode**: Use sensor mode 2 (1920x1080 @ 30fps) for both cameras
   - Eliminates CSI lane mismatch issues
   - Provides excellent frame rate
   - Good balance of quality and performance

3. **Green Overlay**: The green overlay on stereo difference is CORRECT - it shows disparity between cameras, useful for depth perception

## Quick Start Scripts

### High Performance (30 FPS)
```bash
python3 vision_fast.py
```
- Optimized for speed
- Minimal processing
- Async coherence data loading

### Full 1080p Display
```bash
python3 vision_1080p.py
```
- Both cameras at 1920x1080
- Blur detection (press 'b')
- Stereo depth view (press 'd')
- Screenshot (press 's')

### Coherence Integration
```bash
python3 vision_with_display.py
```
- Live coherence overlay
- Reality field visualization
- Sensor contribution display

## GStreamer Pipeline

The working pipeline for both cameras:
```python
f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
f"video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
f"nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! "
f"videoconvert ! video/x-raw, format=BGR ! "
f"appsink drop=true max-buffers=1 sync=false"
```

Key parameters:
- `sensor-mode=2`: Forces 1080p mode
- `drop=true`: Drops old frames
- `max-buffers=1`: Minimal buffering
- `sync=false`: Lowest latency

## Troubleshooting

### Blur on one camera
- Check manual focus rings! They should be at same position
- Enable blur detection with 'b' key to see sharpness scores

### Low frame rate
- Use vision_fast.py instead of vision_with_display.py
- Reduce resolution in pipeline
- Disable depth processing

### Only seeing one camera
- Window may need manual resizing
- Check colored borders (green=left, red=right)
- Verify both cameras initialized in console output

## Performance Metrics
- **Capture**: 1920x1080 @ 30fps
- **Display**: 960x540 per camera (side by side)
- **Latency**: < 50ms with optimized pipeline
- **CPU Usage**: ~30% with both cameras

## Next Steps
- [ ] Integrate stereo depth into coherence engine
- [ ] Add optical flow for motion vectors
- [ ] Implement attention-based ROI selection
- [ ] Create depth-based obstacle detection