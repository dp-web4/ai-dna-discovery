#!/bin/bash
# Launch IMU-Stabilized Binocular Vision

echo "IMU-Stabilized Binocular Vision Launcher"
echo "========================================"
echo ""

# Check directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if we can access the serial port
if [ -r /dev/ttyUSB0 ] && [ -w /dev/ttyUSB0 ]; then
    echo "✓ IMU port accessible"
    echo ""
    echo "Starting stabilized vision system..."
    echo "Controls:"
    echo "  s - Toggle stabilization ON/OFF"
    echo "  r - Reset reference orientation"
    echo "  q - Quit"
    echo ""
    python3 imu_stabilized_vision.py
else
    echo "⚠ Cannot access IMU at /dev/ttyUSB0"
    echo ""
    echo "The system will run without stabilization."
    echo "To enable IMU stabilization:"
    echo "  1. Run with sudo: sudo $0"
    echo "  2. Or fix permissions: sudo chmod 666 /dev/ttyUSB0"
    echo ""
    read -p "Continue without IMU? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        python3 imu_stabilized_vision.py
    fi
fi