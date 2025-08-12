#!/usr/bin/env python3
"""
Test real IMU connection and data format
"""

import serial
import time

print("Testing real IMU on /dev/ttyUSB0...")

try:
    # Open serial connection
    ser = serial.Serial(
        port="/dev/ttyUSB0",
        baudrate=115200,
        timeout=1.0
    )
    
    print(f"✓ Connected to {ser.port} at {ser.baudrate} baud")
    print("\nReading 10 lines of data:\n")
    
    # Read and display some data
    for i in range(10):
        if ser.in_waiting:
            line = ser.readline()
            try:
                # Try to decode as string
                decoded = line.decode('utf-8').strip()
                print(f"Line {i+1}: {decoded}")
            except:
                # Show raw bytes if decode fails
                print(f"Line {i+1} (raw): {line}")
        else:
            print(f"Line {i+1}: (no data)")
            time.sleep(0.1)
    
    ser.close()
    print("\n✓ Test complete")
    
except serial.SerialException as e:
    print(f"✗ Error: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")