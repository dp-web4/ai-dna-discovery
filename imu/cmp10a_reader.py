#!/usr/bin/env python3
"""
Simple CMP10A IMU reader for Jetson Orin Nano
Reads at 9600 baud and displays key values
"""
import serial
import struct
import time
import math

class CMP10AReader:
    def __init__(self, port='/dev/ttyUSB0', baud=9600):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.data = {
            'accel': {'x': 0, 'y': 0, 'z': 0},
            'gyro': {'x': 0, 'y': 0, 'z': 0},
            'angle': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'mag': {'x': 0, 'y': 0, 'z': 0},
            'temp': 0
        }
        
    def parse_packet(self, packet):
        """Parse 11-byte CMP10A packet"""
        if len(packet) != 11 or packet[0] != 0x55:
            return False
            
        # Verify checksum
        checksum = sum(packet[:10]) & 0xFF
        if checksum != packet[10]:
            return False
            
        header2 = packet[1]
        data = packet[2:10]
        
        if header2 == 0x51:  # Acceleration
            ax = struct.unpack('<h', data[0:2])[0] / 32768.0 * 16
            ay = struct.unpack('<h', data[2:4])[0] / 32768.0 * 16
            az = struct.unpack('<h', data[4:6])[0] / 32768.0 * 16
            temp = struct.unpack('<h', data[6:8])[0] / 100.0
            self.data['accel'] = {'x': ax, 'y': ay, 'z': az}
            self.data['temp'] = temp
            
        elif header2 == 0x52:  # Gyroscope
            wx = struct.unpack('<h', data[0:2])[0] / 32768.0 * 2000
            wy = struct.unpack('<h', data[2:4])[0] / 32768.0 * 2000
            wz = struct.unpack('<h', data[4:6])[0] / 32768.0 * 2000
            self.data['gyro'] = {'x': wx, 'y': wy, 'z': wz}
            
        elif header2 == 0x53:  # Angle
            roll = struct.unpack('<h', data[0:2])[0] / 32768.0 * 180
            pitch = struct.unpack('<h', data[2:4])[0] / 32768.0 * 180
            yaw = struct.unpack('<h', data[4:6])[0] / 32768.0 * 180
            self.data['angle'] = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
            
        elif header2 == 0x54:  # Magnetic
            mx = struct.unpack('<h', data[0:2])[0]
            my = struct.unpack('<h', data[2:4])[0]
            mz = struct.unpack('<h', data[4:6])[0]
            self.data['mag'] = {'x': mx, 'y': my, 'z': mz}
            
        return True
        
    def read_loop(self):
        """Simple read loop with basic display"""
        buffer = b''
        last_display = time.time()
        packet_count = 0
        
        print("CMP10A IMU Reader - Press Ctrl+C to stop")
        print("-" * 60)
        
        try:
            while True:
                # Read available data
                if self.ser.in_waiting:
                    buffer += self.ser.read(self.ser.in_waiting)
                
                # Process complete packets
                while len(buffer) >= 11:
                    idx = buffer.find(b'\x55')
                    if idx == -1:
                        buffer = b''
                        break
                    
                    if idx > 0:
                        buffer = buffer[idx:]
                    
                    if len(buffer) >= 11:
                        packet = buffer[:11]
                        if self.parse_packet(packet):
                            packet_count += 1
                        buffer = buffer[11:]
                
                # Display every 100ms
                if time.time() - last_display > 0.1:
                    # Calculate acceleration magnitude
                    acc = self.data['accel']
                    acc_mag = math.sqrt(acc['x']**2 + acc['y']**2 + acc['z']**2)
                    
                    # Simple display (no screen clearing to avoid crashes)
                    print(f"\rAccel: X={acc['x']:6.2f} Y={acc['y']:6.2f} Z={acc['z']:6.2f} |{acc_mag:5.2f}|g  "
                          f"Angle: R={self.data['angle']['roll']:6.1f} P={self.data['angle']['pitch']:6.1f} "
                          f"[{packet_count} pkts]", end='')
                    
                    last_display = time.time()
                    
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        finally:
            self.ser.close()

if __name__ == "__main__":
    reader = CMP10AReader()
    reader.read_loop()