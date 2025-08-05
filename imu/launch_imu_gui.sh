#!/bin/bash
# Launch IMU Visualizer GUI

echo "IMU Visualizer Launcher"
echo "======================"
echo ""

# Check if we can access the serial port
if [ -r /dev/ttyUSB0 ] && [ -w /dev/ttyUSB0 ]; then
    echo "✓ Serial port accessible"
    echo "Launching IMU Visualizer..."
    echo "Close the window to exit cleanly"
    python3 $(dirname "$0")/imu_visualizer.py
else
    echo "✗ Cannot access /dev/ttyUSB0"
    echo ""
    echo "Options:"
    echo "1. Run with sudo (recommended):"
    echo "   sudo $(dirname "$0")/launch_imu_gui.sh"
    echo ""
    echo "2. Or fix permissions temporarily:"
    echo "   sudo chmod 666 /dev/ttyUSB0"
    echo ""
    
    # If not running as root, offer to run with sudo
    if [ "$EUID" -ne 0 ]; then
        read -p "Run with sudo now? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            sudo python3 $(dirname "$0")/imu_visualizer.py
        fi
    else
        # Already root, just run it
        python3 $(dirname "$0")/imu_visualizer.py
    fi
fi