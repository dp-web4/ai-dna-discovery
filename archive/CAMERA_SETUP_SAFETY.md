# Jetson Orin Nano Camera Setup Safety Guide

## 🚨 CRITICAL LESSONS LEARNED
1. **NEVER change UEFI settings** (Device Tree → ACPI = BRICK!)
2. **Power delivery issues** - Cameras draw significant current at init
3. **Start with ONE camera on CSI0** - Don't connect both initially
4. **Always backup device tree** before any camera changes
5. **System error popups are WARNING SIGNS** - Stop immediately

## Pre-Connection Checklist

### 1. Current System Backup
```bash
# Backup current device tree
sudo dtc -I fs -O dts -o ~/device_tree_backup_$(date +%Y%m%d_%H%M%S).dts /proc/device-tree

# Save current working configuration
sudo cp /boot/extlinux/extlinux.conf ~/extlinux_backup_$(date +%Y%m%d_%H%M%S).conf

# Document current kernel modules
lsmod > ~/kernel_modules_backup_$(date +%Y%m%d_%H%M%S).txt

# Save dmesg for baseline
dmesg > ~/dmesg_baseline_$(date +%Y%m%d_%H%M%S).txt
```

### 2. Power Considerations
- Jetson powered via USB-C (confirmed working)
- Camera will draw power from CSI interface
- **WARNING**: Both cameras together may exceed power budget
- Monitor for undervoltage warnings in dmesg

### 3. Physical Connection Plan
1. **Power OFF completely** (not just shutdown)
2. Connect **ONE** IMX219 camera to **CSI0 port only**
3. Ensure ribbon cable is:
   - Blue side facing toward pins
   - Firmly seated but not forced
   - Not twisted or bent sharply
4. Leave CSI1 empty for now

## Post-Connection Test Sequence

### 1. Initial Power-On
```bash
# Monitor boot messages
dmesg -w  # In one terminal

# In another terminal after boot:
# Check if camera detected
ls /dev/video*

# Check device tree for camera
ls /proc/device-tree/cam_i2cmux/i2c@*
```

### 2. Safe Camera Detection
```bash
# Use v4l2-ctl (safer than nvgstcapture initially)
v4l2-ctl --list-devices

# If camera appears, check capabilities
v4l2-ctl -d /dev/video0 --all

# Check for i2c devices
sudo i2cdetect -y -r 0
sudo i2cdetect -y -r 1
```

### 3. Minimal Test (if detected)
```bash
# Simple frame grab (no preview window)
v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=480 --stream-mmap --stream-count=1 --stream-to=test.raw

# Only if above works, try gstreamer
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! 'video/x-raw(memory:NVMM),width=640,height=480' ! nvjpegenc ! filesink location=test.jpg
```

## 🛑 STOP SIGNS
Immediately power off if you see:
- System error popups
- Kernel panics or oops
- Multiple i2c timeout messages
- Power warnings in dmesg
- Camera not detected after 2 reboots

## Recovery Plan
If issues occur:
1. Power off immediately
2. Disconnect camera
3. Boot without camera to verify system still works
4. Check logs: `dmesg | grep -E "imx219|camera|csi|i2c"`
5. Do NOT attempt UEFI changes

## Important Notes
- Orin uses **tegra234** (not tegra210)
- Different camera stack than original Nano
- JetPack 6.2.1 has specific camera requirements
- Default device tree should support IMX219

## Next Steps After Success
Only after single camera works reliably:
1. Test different resolutions
2. Verify stable operation over time
3. Consider second camera (with power monitoring)
4. Never modify UEFI settings

Remember: **Working Jetson > Working Camera**