#!/usr/bin/env python3
"""
Test script to detect and read IMU data
"""
import serial
import time
import sys

# Common baud rates for IMUs
BAUD_RATES = [9600, 19200, 38400, 57600, 115200, 230400]

def test_serial_port(port, baud):
    """Test if we can read data at given baud rate"""
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(0.1)  # Give device time to init
        
        # Try reading some data
        data = ser.read(100)
        ser.close()
        
        if data:
            return True, data
        return False, None
    except Exception as e:
        return False, str(e)

def detect_imu_protocol(port):
    """Try to detect IMU protocol"""
    print(f"Testing port {port}...")
    
    for baud in BAUD_RATES:
        success, data = test_serial_port(port, baud)
        if success:
            print(f"\n✓ Found data at {baud} baud:")
            print(f"Raw bytes (first 50): {data[:50]}")
            print(f"ASCII attempt: {data[:50].decode('ascii', errors='ignore')}")
            
            # Check for common IMU protocols
            if b'$' in data:
                print("→ Possible NMEA-style protocol (GPS/IMU)")
            if b'\xAA' in data or b'\x55' in data:
                print("→ Possible binary protocol")
            if b'MPU' in data or b'BNO' in data or b'IMU' in data:
                print("→ Device identifier found")
                
            # Try to parse as different formats
            try:
                lines = data.decode('utf-8').strip().split('\n')
                print(f"→ UTF-8 lines: {lines[:3]}")
            except:
                pass
                
            return baud, data
        else:
            print(f"✗ {baud} baud: No data")
    
    return None, None

if __name__ == "__main__":
    port = "/dev/ttyUSB0"
    if len(sys.argv) > 1:
        port = sys.argv[1]
    
    baud, data = detect_imu_protocol(port)
    
    if baud:
        print(f"\n🎯 IMU likely communicating at {baud} baud")
        print("\nTo read continuously, try:")
        print(f"  sudo screen {port} {baud}")
        print(f"  sudo minicom -D {port} -b {baud}")
    else:
        print("\n❌ Could not detect IMU data. Device might need:")
        print("  - Different baud rate")
        print("  - Initialization command")
        print("  - Different protocol/format")