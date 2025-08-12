#!/usr/bin/env python3
"""
Direct test of Yahboom IMU to verify it's working
"""
import sys
import time
sys.path.append('/home/sprout/ai-workspace/private-context/ai-dna-discovery/imu')
from yahboom_cmp10a import YahboomCMP10A

print("Testing Yahboom CMP10A IMU...")
imu = YahboomCMP10A()

# Storage for latest values
latest = {
    'accel': None,
    'gyro': None,
    'angle': None,
    'mag': None
}

print("\nReading IMU data for 5 seconds...")
print("Move the IMU to see values change!\n")

start_time = time.time()
packet_count = 0

while time.time() - start_time < 5:
    # Read data
    data = imu.ser.read(imu.ser.in_waiting or 1)
    if data:
        imu.buffer += data
        
    # Process packets
    while len(imu.buffer) >= 11:
        idx = imu.buffer.find(b'\x55')
        if idx == -1:
            imu.buffer = b''
            break
            
        if len(imu.buffer) >= idx + 11:
            packet = imu.buffer[idx:idx + 11]
            imu.buffer = imu.buffer[idx + 11:]
            
            parsed = imu.parse_packet(packet)
            if parsed:
                packet_count += 1
                latest[parsed['type']] = parsed
                
                # Print current values every 10 packets
                if packet_count % 10 == 0:
                    print(f"Packets: {packet_count}")
                    if latest['accel']:
                        print(f"  Accel: X={latest['accel']['x']:+7.3f} Y={latest['accel']['y']:+7.3f} Z={latest['accel']['z']:+7.3f}")
                    if latest['gyro']:
                        print(f"  Gyro:  X={latest['gyro']['x']:+7.1f} Y={latest['gyro']['y']:+7.1f} Z={latest['gyro']['z']:+7.1f}")
                    if latest['angle']:
                        print(f"  Angle: R={latest['angle']['roll']:+7.1f} P={latest['angle']['pitch']:+7.1f} Y={latest['angle']['yaw']:+7.1f}")
                    print()
        else:
            break

print(f"\nTotal packets received: {packet_count}")
print(f"Rate: {packet_count/5:.1f} packets/sec")

imu.ser.close()