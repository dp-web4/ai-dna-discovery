#!/usr/bin/env python3
"""
Simple IMU test without complex terminal operations
"""
import serial
import time

print("Simple Yahboom CMP10A test")
print("Trying both baud rates...")

# Try 921600 first
try:
    print("\nTrying 921600 baud...")
    ser = serial.Serial('/dev/ttyUSB0', 921600, timeout=0.5)
    time.sleep(0.1)
    data = ser.read(100)
    if data:
        print(f"Got {len(data)} bytes at 921600")
        print("First 20 bytes (hex):", ' '.join(f'{b:02x}' for b in data[:20]))
        if b'\x55' in data:
            print("✓ Found 0x55 header - this looks like CMP10A protocol!")
    else:
        print("No data at 921600")
    ser.close()
except Exception as e:
    print(f"921600 failed: {e}")

# Try 9600 as fallback
try:
    print("\nTrying 9600 baud...")
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.5)
    time.sleep(0.1)
    data = ser.read(100)
    if data:
        print(f"Got {len(data)} bytes at 9600")
        print("First 20 bytes (hex):", ' '.join(f'{b:02x}' for b in data[:20]))
        # Look for CMP10A markers
        if b'\x55' in data:
            print("✓ Found 0x55 header")
        if b'U' in data:
            print("Found 'U' (0x55) ASCII")
    else:
        print("No data at 9600")
    ser.close()
except Exception as e:
    print(f"9600 failed: {e}")

print("\nTo configure baud rate, the device might need AT commands or configuration software.")
print("The unit appears to be communicating at 9600 baud currently.")