#!/usr/bin/env python3
"""
IMU reader for binary protocol detection
"""
import serial
import struct
import time
import sys

def analyze_binary_pattern(data):
    """Analyze binary data to find patterns"""
    # Look for common IMU packet headers
    headers = {
        b'\xAA\x55': 'MPU6050/9250 style',
        b'\x55\xAA': 'Reversed MPU style', 
        b'\xFB': 'Possible frame start',
        b'\x42\x4D': 'BNO055',
        b'\x24': 'NMEA style'
    }
    
    for header, name in headers.items():
        if header in data:
            print(f"Found {name} header: {header.hex()}")
            
    # Look for repeating patterns
    for i in range(1, min(20, len(data)//2)):
        if data[:i] == data[i:2*i]:
            print(f"Repeating pattern of {i} bytes: {data[:i].hex()}")
            
def parse_imu_data(ser, duration=5):
    """Try to parse IMU data for given duration"""
    print(f"\nReading IMU data for {duration} seconds...\n")
    
    start_time = time.time()
    buffer = b''
    packet_sizes = {}
    
    while time.time() - start_time < duration:
        data = ser.read(ser.in_waiting or 1)
        if data:
            buffer += data
            
            # Look for packet boundaries (0xFB seems common)
            if b'\xfb' in buffer:
                parts = buffer.split(b'\xfb')
                for i, part in enumerate(parts[:-1]):
                    if len(part) > 0:
                        size = len(part)
                        packet_sizes[size] = packet_sizes.get(size, 0) + 1
                        
                        # Try to parse if consistent size
                        if size == 11:  # Common IMU packet size
                            try:
                                # Try parsing as different formats
                                values = struct.unpack('<HHHHbbb', part[:11])
                                print(f"Possible 11-byte packet: {values}")
                            except:
                                pass
                                
                buffer = b'\xfb' + parts[-1]
    
    print(f"\nPacket size distribution:")
    for size, count in sorted(packet_sizes.items()):
        print(f"  {size} bytes: {count} packets")
        
    return buffer

def continuous_read(port, baud):
    """Continuously read and display IMU data"""
    ser = serial.Serial(port, baud, timeout=0.1)
    print(f"Reading from {port} at {baud} baud...")
    print("Press Ctrl+C to stop\n")
    
    # First analyze the pattern
    initial_data = ser.read(200)
    analyze_binary_pattern(initial_data)
    
    # Then try to parse
    try:
        remaining = parse_imu_data(ser, 3)
        
        print("\nRaw hex dump of last data:")
        for i in range(0, min(len(remaining), 100), 16):
            hex_str = ' '.join(f'{b:02x}' for b in remaining[i:i+16])
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in remaining[i:i+16])
            print(f"{i:04x}: {hex_str:<48} {ascii_str}")
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        ser.close()

if __name__ == "__main__":
    port = "/dev/ttyUSB0"
    baud = 9600
    
    if len(sys.argv) > 1:
        port = sys.argv[1]
    if len(sys.argv) > 2:
        baud = int(sys.argv[2])
        
    continuous_read(port, baud)