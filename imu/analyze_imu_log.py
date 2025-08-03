#!/usr/bin/env python3
"""
Analyze IMU log to understand packet structure
"""
import sys

def analyze_log(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    
    print(f"Analyzing {filename}")
    print(f"Total bytes: {len(data)}")
    
    # Look for 0x55 headers
    headers = []
    for i in range(len(data)):
        if data[i] == 0x55:
            headers.append(i)
    
    print(f"\nFound {len(headers)} potential 0x55 headers")
    
    # Analyze packet sizes
    if len(headers) > 1:
        sizes = []
        for i in range(len(headers)-1):
            size = headers[i+1] - headers[i]
            sizes.append(size)
        
        # Count size frequencies
        size_freq = {}
        for s in sizes:
            size_freq[s] = size_freq.get(s, 0) + 1
        
        print("\nPacket size distribution:")
        for size, count in sorted(size_freq.items()):
            print(f"  {size} bytes: {count} times")
    
    # Show some example packets
    print("\nExample packets (first 5):")
    for i in range(min(5, len(headers)-1)):
        start = headers[i]
        end = headers[i+1] if i+1 < len(headers) else min(start+20, len(data))
        packet = data[start:end]
        
        print(f"\nPacket {i+1} at offset {start}:")
        hex_str = ' '.join(f'{b:02x}' for b in packet)
        print(f"  Hex: {hex_str}")
        
        # If it's 11 bytes (standard CMP10A packet)
        if len(packet) == 11:
            header2 = packet[1]
            checksum_calc = sum(packet[:10]) & 0xFF
            checksum_pkt = packet[10]
            
            types = {0x51: 'Accel', 0x52: 'Gyro', 0x53: 'Angle', 0x54: 'Mag', 0x56: 'Baro'}
            pkt_type = types.get(header2, f'Unknown(0x{header2:02x})')
            
            print(f"  Type: {pkt_type}")
            print(f"  Checksum: calc={checksum_calc:02x}, packet={checksum_pkt:02x}, {'OK' if checksum_calc == checksum_pkt else 'BAD'}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_log(sys.argv[1])
    else:
        analyze_log("/data/sensor-data/imu_log_20250802_183345.bin")