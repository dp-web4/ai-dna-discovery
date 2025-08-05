#!/bin/bash
# Fix IMU serial port permissions

echo "IMU Permission Helper"
echo "===================="
echo ""

# Check if user is in dialout group
if groups $USER | grep -q '\bdialout\b'; then
    echo "✓ You are in the dialout group"
    echo ""
    echo "However, you need to logout and login again for it to take effect."
    echo "Until then, you can:"
    echo ""
    echo "1. Run tools with sudo:"
    echo "   sudo python3 imu_visualizer.py"
    echo ""
    echo "2. Or temporarily fix permissions (until reboot):"
    echo "   sudo chmod 666 /dev/ttyUSB0"
    echo ""
else
    echo "✗ You are NOT in the dialout group"
    echo ""
    echo "Adding you to dialout group..."
    sudo usermod -a -G dialout $USER
    echo "✓ Added to dialout group"
    echo ""
    echo "IMPORTANT: You must logout and login again for this to take effect!"
    echo ""
    echo "Until then, you can:"
    echo "1. Run tools with sudo:"
    echo "   sudo python3 imu_visualizer.py"
    echo ""
    echo "2. Or temporarily fix permissions (until reboot):"
    echo "   sudo chmod 666 /dev/ttyUSB0"
fi

echo ""
echo "Current /dev/ttyUSB0 permissions:"
ls -la /dev/ttyUSB0 2>/dev/null || echo "Device not found"