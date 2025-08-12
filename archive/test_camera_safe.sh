#!/bin/bash
# Safe camera testing script for Jetson Orin Nano
# Follows lessons learned from previous attempts

echo "🎥 Jetson Orin Nano Camera Test (Safe Mode)"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running with appropriate permissions
if [ "$EUID" -eq 0 ]; then 
   echo -e "${YELLOW}Note: Running as root${NC}"
fi

# Function to check for danger signs
check_system_health() {
    echo "🏥 Checking system health..."
    
    # Check for kernel errors
    if dmesg | tail -50 | grep -E "kernel panic|Oops|BUG:" > /dev/null; then
        echo -e "${RED}❌ CRITICAL: Kernel errors detected! Stop immediately!${NC}"
        exit 1
    fi
    
    # Check for i2c timeouts
    if dmesg | tail -50 | grep -E "i2c.*timeout" > /dev/null; then
        echo -e "${YELLOW}⚠️  Warning: I2C timeouts detected${NC}"
    fi
    
    echo -e "${GREEN}✅ System health OK${NC}"
}

# Initial system check
check_system_health

echo ""
echo "📊 Step 1: Checking for video devices..."
if ls /dev/video* 2>/dev/null; then
    echo -e "${GREEN}Found video devices:${NC}"
    ls -la /dev/video*
else
    echo -e "${YELLOW}No video devices found (camera may not be connected/detected)${NC}"
    echo ""
    echo "🔍 Checking dmesg for camera messages..."
    dmesg | grep -E "imx219|imx|camera|tegra-capture" | tail -10
    exit 1
fi

echo ""
echo "📊 Step 2: V4L2 device check..."
v4l2-ctl --list-devices

echo ""
echo "📊 Step 3: Checking I2C buses for camera..."
echo "Camera should appear at address 0x10 on one of these buses:"
for bus in 0 1 2 7 8 9; do
    echo -n "Bus $bus: "
    sudo i2cdetect -y -r $bus 2>/dev/null | grep -E "10|UU" || echo "nothing found"
done

# Check system health again
check_system_health

echo ""
echo "📊 Step 4: Camera capabilities (if detected)..."
if [ -e /dev/video0 ]; then
    echo "Checking /dev/video0:"
    v4l2-ctl -d /dev/video0 --all 2>/dev/null | grep -E "Driver name|Card type|Bus info|Driver version|Width/Height|Pixel Format" || echo "Could not query device"
fi

echo ""
echo "🧪 Step 5: Minimal capture test..."
echo -e "${YELLOW}This will attempt a single frame capture without display${NC}"
read -p "Continue with capture test? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check system health before capture
    check_system_health
    
    # Try simple v4l2 capture first (safest)
    echo "Attempting v4l2 capture..."
    if v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG --stream-mmap --stream-count=1 --stream-to=test_v4l2.jpg 2>/dev/null; then
        echo -e "${GREEN}✅ V4L2 capture successful!${NC}"
        ls -la test_v4l2.jpg
    else
        echo -e "${YELLOW}V4L2 capture failed, trying nvarguscamerasrc...${NC}"
        
        # Try gstreamer with nvargus (Jetson specific)
        if gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! \
           'video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1' ! \
           nvjpegenc ! filesink location=test_nvargus.jpg 2>/dev/null; then
            echo -e "${GREEN}✅ Nvargus capture successful!${NC}"
            ls -la test_nvargus.jpg
        else
            echo -e "${RED}❌ Both capture methods failed${NC}"
            echo "Checking dmesg for errors..."
            dmesg | tail -20 | grep -E "imx219|camera|csi|video"
        fi
    fi
    
    # Final system health check
    check_system_health
fi

echo ""
echo "📋 Summary:"
echo "==========="
if [ -e test_v4l2.jpg ] || [ -e test_nvargus.jpg ]; then
    echo -e "${GREEN}✅ Camera test completed successfully!${NC}"
    echo "Test images created:"
    ls -la test*.jpg 2>/dev/null
else
    echo -e "${YELLOW}⚠️  Camera test incomplete or failed${NC}"
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "1. Ensure camera ribbon cable is properly connected to CSI0"
    echo "2. Blue side of ribbon should face the pins"
    echo "3. Check power - camera needs adequate current"
    echo "4. Try a reboot if this is first connection"
    echo "5. NEVER change UEFI settings to fix camera issues"
fi

echo ""
echo "📁 Logs saved for analysis:"
echo "- dmesg output: dmesg > camera_debug.log"
echo "- I2C state: saved above"
echo "- Video device info: ls -la /dev/video*"

# Save debug info
dmesg | grep -E "imx219|camera|csi|tegra-capture|video" > camera_debug.log
echo "" >> camera_debug.log
echo "I2C Detection Results:" >> camera_debug.log
for bus in 0 1 2 7 8 9; do
    echo "Bus $bus:" >> camera_debug.log
    sudo i2cdetect -y -r $bus 2>/dev/null >> camera_debug.log
done

echo ""
echo "🏁 Test complete. Debug log saved to camera_debug.log"