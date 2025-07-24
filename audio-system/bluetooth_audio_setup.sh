#!/bin/bash
# Bluetooth Audio Setup Script for Sprout

echo "=== Bluetooth Audio Setup for Voice Conversations ==="
echo ""
echo "1. Make sure your Bluetooth headset is in pairing mode"
echo "2. Running interactive Bluetooth setup..."
echo ""

# Start bluetoothctl in interactive mode
echo "Commands to use in bluetoothctl:"
echo "  power on         - Turn on Bluetooth"
echo "  scan on          - Start scanning"
echo "  devices          - List found devices"
echo "  pair <ADDRESS>   - Pair with device"
echo "  connect <ADDRESS> - Connect to device"
echo "  trust <ADDRESS>  - Trust device for auto-connect"
echo "  exit             - Exit when done"
echo ""
echo "Look for your headset name in the device list!"
echo ""

bluetoothctl