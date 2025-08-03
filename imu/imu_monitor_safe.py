#!/usr/bin/env python3
"""
Safe IMU monitor - no fancy terminal operations
Writes to a file that can be tailed or viewed separately
"""
import serial
import struct
import time
import sys
import signal

class SafeIMUMonitor:
    def __init__(self, port='/dev/ttyUSB0', baud=115200, output_file='/tmp/imu_data.txt'):
        self.port = port
        self.baud = baud
        self.output_file = output_file
        self.running = True
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Data storage
        self.data = {
            'accel': {'x': 0, 'y': 0, 'z': 0},
            'gyro': {'x': 0, 'y': 0, 'z': 0},
            'angle': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'temp': 0
        }
        
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\nShutting down gracefully...")
        self.running = False
        
    def parse_packet(self, packet):
        """Parse IMU packet"""
        header2 = packet[1]
        data = packet[2:10]
        
        if header2 == 0x51:  # Acceleration
            self.data['accel']['x'] = struct.unpack('<h', data[0:2])[0] / 32768.0 * 16
            self.data['accel']['y'] = struct.unpack('<h', data[2:4])[0] / 32768.0 * 16
            self.data['accel']['z'] = struct.unpack('<h', data[4:6])[0] / 32768.0 * 16
            self.data['temp'] = struct.unpack('<h', data[6:8])[0] / 100.0
            
        elif header2 == 0x52:  # Gyroscope
            self.data['gyro']['x'] = struct.unpack('<h', data[0:2])[0] / 32768.0 * 2000
            self.data['gyro']['y'] = struct.unpack('<h', data[2:4])[0] / 32768.0 * 2000
            self.data['gyro']['z'] = struct.unpack('<h', data[4:6])[0] / 32768.0 * 2000
            
        elif header2 == 0x53:  # Angle
            self.data['angle']['roll'] = struct.unpack('<h', data[0:2])[0] / 32768.0 * 180
            self.data['angle']['pitch'] = struct.unpack('<h', data[2:4])[0] / 32768.0 * 180
            self.data['angle']['yaw'] = struct.unpack('<h', data[4:6])[0] / 32768.0 * 180
            
    def monitor(self):
        """Monitor IMU and write to file"""
        print(f"Safe IMU Monitor")
        print(f"Port: {self.port} @ {self.baud} baud")
        print(f"Output: {self.output_file}")
        print(f"")
        print(f"In another terminal, run:")
        print(f"  tail -f {self.output_file}")
        print(f"")
        print(f"Press Ctrl+C to stop")
        print("-" * 50)
        
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.1)
            buffer = b''
            packet_count = 0
            start_time = time.time()
            last_write = time.time()
            
            with open(self.output_file, 'w') as f:
                f.write("IMU Data Stream\n")
                f.write("===============\n\n")
                
            while self.running:
                # Read data
                if ser.in_waiting:
                    buffer += ser.read(ser.in_waiting)
                    
                # Process packets
                while len(buffer) >= 11:
                    idx = buffer.find(b'\x55')
                    if idx == -1:
                        buffer = b''
                        break
                        
                    buffer = buffer[idx:]
                    
                    if len(buffer) >= 11:
                        packet = buffer[:11]
                        
                        # Verify checksum
                        if sum(packet[:10]) & 0xFF == packet[10]:
                            self.parse_packet(packet)
                            packet_count += 1
                            
                        buffer = buffer[11:]
                        
                # Write to file every 100ms
                if time.time() - last_write > 0.1:
                    elapsed = time.time() - start_time
                    rate = packet_count / elapsed if elapsed > 0 else 0
                    
                    # Write to file
                    with open(self.output_file, 'w') as f:
                        f.write(f"IMU Data - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Rate: {rate:.1f} Hz | Packets: {packet_count}\n")
                        f.write("=" * 60 + "\n\n")
                        
                        f.write("Acceleration (g):\n")
                        f.write(f"  X: {self.data['accel']['x']:7.3f}\n")
                        f.write(f"  Y: {self.data['accel']['y']:7.3f}\n")
                        f.write(f"  Z: {self.data['accel']['z']:7.3f}\n")
                        
                        accel_mag = (self.data['accel']['x']**2 + 
                                   self.data['accel']['y']**2 + 
                                   self.data['accel']['z']**2)**0.5
                        f.write(f"  Magnitude: {accel_mag:7.3f}\n\n")
                        
                        f.write("Gyroscope (°/s):\n")
                        f.write(f"  X: {self.data['gyro']['x']:7.1f}\n")
                        f.write(f"  Y: {self.data['gyro']['y']:7.1f}\n")
                        f.write(f"  Z: {self.data['gyro']['z']:7.1f}\n\n")
                        
                        f.write("Orientation (°):\n")
                        f.write(f"  Roll:  {self.data['angle']['roll']:7.1f}\n")
                        f.write(f"  Pitch: {self.data['angle']['pitch']:7.1f}\n")
                        f.write(f"  Yaw:   {self.data['angle']['yaw']:7.1f}\n\n")
                        
                        f.write(f"Temperature: {self.data.get('temp', 0):5.1f}°C\n")
                        
                    # Also print summary to console
                    print(f"\rRate: {rate:5.1f}Hz | "
                          f"Roll: {self.data['angle']['roll']:6.1f}° | "
                          f"Pitch: {self.data['angle']['pitch']:6.1f}° | "
                          f"Yaw: {self.data['angle']['yaw']:6.1f}°", end='')
                    
                    last_write = time.time()
                    
        except serial.SerialException as e:
            print(f"\nSerial error: {e}")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            if 'ser' in locals():
                ser.close()
            print("\nMonitor stopped.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Safe IMU Monitor')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--output', default='/tmp/imu_data.txt', help='Output file')
    
    args = parser.parse_args()
    
    monitor = SafeIMUMonitor(args.port, args.baud, args.output)
    monitor.monitor()

if __name__ == "__main__":
    main()