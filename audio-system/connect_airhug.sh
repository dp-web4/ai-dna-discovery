#!/bin/bash
# Connect to AIRHUG Bluetooth headset

echo "Searching for AIRHUG device..."

# Power on and scan
bluetoothctl power on
sleep 1
bluetoothctl agent on
sleep 1

# Scan for devices and find AIRHUG
echo "Scanning for Bluetooth devices (15 seconds)..."
bluetoothctl scan on &
SCAN_PID=$!
sleep 15
kill $SCAN_PID 2>/dev/null

# Get AIRHUG address
AIRHUG_ADDR=$(bluetoothctl devices | grep -i "AIRHUG" | awk '{print $2}')

if [ -z "$AIRHUG_ADDR" ]; then
    echo "AIRHUG device not found. Make sure it's in pairing mode."
    echo "Running interactive mode..."
    bluetoothctl
else
    echo "Found AIRHUG at: $AIRHUG_ADDR"
    echo "Attempting to pair and connect..."
    
    bluetoothctl pair $AIRHUG_ADDR
    sleep 2
    bluetoothctl connect $AIRHUG_ADDR
    sleep 2
    bluetoothctl trust $AIRHUG_ADDR
    
    echo ""
    echo "Checking connection status..."
    bluetoothctl info $AIRHUG_ADDR | grep -E "Name|Connected|Paired"
fi

echo ""
echo "Testing audio devices..."
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
print('\nAvailable audio devices:')
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if 'bluetooth' in info['name'].lower() or 'bluez' in info['name'].lower() or 'airhug' in info['name'].lower():
        print(f'🎧 BT Device {i}: {info[\"name\"]} (in:{info[\"maxInputChannels\"]}, out:{info[\"maxOutputChannels\"]})')
p.terminate()
" 2>/dev/null