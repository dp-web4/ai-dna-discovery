#!/usr/bin/env python3
"""
IMU decoder for the detected binary protocol
Packet structure appears to be:
- 0xFB header
- 4 or 38 byte packets
"""
import serial
import struct
import time
import math

class IMUDecoder:
    def __init__(self, port='/dev/ttyUSB0', baud=9600):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.buffer = b''
        
    def read_packets(self):
        """Read and decode IMU packets"""
        while True:
            # Read available data
            data = self.ser.read(self.ser.in_waiting or 1)
            if data:
                self.buffer += data
                
            # Process buffer for complete packets
            while b'\xfb' in self.buffer:
                start = self.buffer.find(b'\xfb')
                
                # Look for next packet start
                next_start = self.buffer.find(b'\xfb', start + 1)
                
                if next_start != -1:
                    packet = self.buffer[start:next_start]
                    self.buffer = self.buffer[next_start:]
                    
                    if len(packet) == 5:  # 0xFB + 4 bytes
                        self.decode_short_packet(packet)
                    elif len(packet) == 39:  # 0xFB + 38 bytes
                        self.decode_long_packet(packet)
                else:
                    # Wait for more data
                    break
                    
    def decode_short_packet(self, packet):
        """Decode 5-byte packet"""
        try:
            # Skip 0xFB header
            data = packet[1:]
            # Try as 4-byte float or 2 int16s
            if len(data) == 4:
                # As float
                value_f = struct.unpack('<f', data)[0]
                # As 2 int16
                values_i = struct.unpack('<hh', data)
                print(f"Short packet: float={value_f:.3f}, int16s={values_i}")
        except Exception as e:
            print(f"Short packet decode error: {e}")
            
    def decode_long_packet(self, packet):
        """Decode 39-byte packet - likely full IMU data"""
        try:
            # Skip 0xFB header
            data = packet[1:]
            
            # Common IMU data structure guesses:
            # 1. 9-axis: 3x accel, 3x gyro, 3x mag (9 floats = 36 bytes)
            if len(data) >= 36:
                values = struct.unpack('<9f', data[:36])
                ax, ay, az = values[0:3]
                gx, gy, gz = values[3:6]
                mx, my, mz = values[6:9]
                
                print(f"IMU Data:")
                print(f"  Accel: X={ax:7.3f} Y={ay:7.3f} Z={az:7.3f}")
                print(f"  Gyro:  X={gx:7.3f} Y={gy:7.3f} Z={gz:7.3f}")
                print(f"  Mag:   X={mx:7.3f} Y={my:7.3f} Z={mz:7.3f}")
                
                # Calculate magnitudes
                accel_mag = math.sqrt(ax*ax + ay*ay + az*az)
                print(f"  Accel magnitude: {accel_mag:.3f}")
                print("")
                
            # Alternative: 6 int16 values + quaternion
            elif len(data) >= 20:
                values = struct.unpack('<6h4f', data[:20])
                print(f"Alt format: {values}")
                
        except Exception as e:
            # Hex dump if can't decode
            hex_str = ' '.join(f'{b:02x}' for b in data[:20])
            print(f"Long packet ({len(data)}B): {hex_str}...")

def main():
    print("IMU Decoder Starting...")
    print("Assuming 9-axis IMU with accel/gyro/mag data")
    print("Press Ctrl+C to stop\n")
    
    decoder = IMUDecoder()
    
    try:
        decoder.read_packets()
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        decoder.ser.close()

if __name__ == "__main__":
    main()