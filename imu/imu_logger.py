#!/usr/bin/env python3
"""
Basic IMU data logger for Yahboom CMP10A
Saves raw data to file for analysis
"""
import serial
import time
import datetime

def log_imu_data(duration=10):
    """Log IMU data to file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/data/sensor-data/imu_log_{timestamp}.bin"
    
    print(f"Logging IMU data for {duration} seconds to {log_file}")
    
    # Try 9600 baud since that's what we detected
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.1)
    
    with open(log_file, 'wb') as f:
        start_time = time.time()
        bytes_read = 0
        
        while time.time() - start_time < duration:
            data = ser.read(ser.in_waiting or 1)
            if data:
                f.write(data)
                bytes_read += len(data)
                print(f"\rRead {bytes_read} bytes...", end='')
                
    ser.close()
    print(f"\nLogged {bytes_read} bytes to {log_file}")
    
    # Quick analysis
    with open(log_file, 'rb') as f:
        data = f.read()
        
    print(f"\nQuick analysis:")
    print(f"- Total bytes: {len(data)}")
    header_byte = b'\x55'
    print(f"- Contains 0x55: {header_byte in data}")
    print(f"- Contains 'U': {b'U' in data}")
    
    # Show first few bytes
    if len(data) > 0:
        print(f"- First 20 bytes: {' '.join(f'{b:02x}' for b in data[:20])}")

if __name__ == "__main__":
    log_imu_data(5)