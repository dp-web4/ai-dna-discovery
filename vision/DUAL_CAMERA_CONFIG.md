# Dual Camera Configuration

## Setup Complete (July 27, 2025)

### Configuration Applied
- Changed from single IMX219-A to dual IMX219 overlay
- Updated `/boot/extlinux/extlinux.conf`
- Overlay: `/boot/tegra234-p3767-camera-p3768-imx219-dual.dtbo`

### Before Reboot
- Backup created: ~/extlinux_backup_dual_camera_*.conf
- Single camera working on CSI0 (/dev/video0)
- Second camera physically connected to CSI1

### After Reboot - CONFIRMED WORKING ✅
- /dev/video0 (CSI0 - Camera 0) - imx219 9-0010
- /dev/video1 (CSI1 - Camera 1) - imx219 10-0010
- Both cameras tested and operational!

### Test Commands
```bash
# List devices
v4l2-ctl --list-devices

# Test camera 0
v4l2-ctl --device=/dev/video0 --stream-mmap --stream-to=cam0.raw --stream-count=1

# Test camera 1
v4l2-ctl --device=/dev/video1 --stream-mmap --stream-to=cam1.raw --stream-count=1

# GStreamer test for both
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! fakesink
gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! fakesink
```

### Stereo Vision Pipeline
```bash
# Display both cameras side by side
gst-launch-1.0 \
  nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=640,height=480' ! nvvidconv ! queue ! comp.sink_0 \
  nvarguscamerasrc sensor-id=1 ! 'video/x-raw(memory:NVMM),width=640,height=480' ! nvvidconv ! queue ! comp.sink_1 \
  nvcompositor name=comp sink_0::xpos=0 sink_1::xpos=640 ! nvegltransform ! nveglglessink
```

## Notes
- Both cameras are IMX219 sensors
- Each can do 1280x720 @ 60fps or higher resolutions at lower fps
- Power draw increases with both cameras active
- Monitor dmesg for any power warnings