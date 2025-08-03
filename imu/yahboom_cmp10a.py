#!/usr/bin/env python3
"""
Yahboom CMP10A 10-DOF IMU decoder
10-DOF: 3-axis accel, 3-axis gyro, 3-axis mag, 1 barometer
Default baud rate: 921600
"""
import serial
import struct
import time
import sys

class YahboomCMP10A:
    def __init__(self, port='/dev/ttyUSB0', baud=921600):
        """Initialize IMU connection"""
        print(f"Connecting to Yahboom CMP10A at {port} ({baud} baud)...")
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            print("Connected!")
        except Exception as e:
            print(f"Failed to connect: {e}")
            print("Trying fallback baud rate 9600...")
            self.ser = serial.Serial(port, 9600, timeout=0.1)
            print("Connected at 9600 baud")
            
        self.buffer = b''
        
        # CMP10A protocol info
        self.HEADER1 = 0x55
        self.HEADER2_OPTIONS = {
            0x51: 'acceleration',
            0x52: 'angular_velocity', 
            0x53: 'angle',
            0x54: 'magnetic',
            0x56: 'pressure_altitude'
        }
        
    def parse_packet(self, packet):
        """Parse a complete 11-byte packet"""
        if len(packet) != 11:
            return None
            
        # Check header and checksum
        if packet[0] != 0x55:
            return None
            
        # Calculate checksum (sum of first 10 bytes)
        checksum = sum(packet[:10]) & 0xFF
        if checksum != packet[10]:
            return None
            
        header2 = packet[1]
        data = packet[2:10]
        
        if header2 == 0x51:  # Acceleration
            ax = struct.unpack('<h', data[0:2])[0] / 32768.0 * 16  # ±16g range
            ay = struct.unpack('<h', data[2:4])[0] / 32768.0 * 16
            az = struct.unpack('<h', data[4:6])[0] / 32768.0 * 16
            temp = struct.unpack('<h', data[6:8])[0] / 100.0
            return {'type': 'accel', 'x': ax, 'y': ay, 'z': az, 'temp': temp}
            
        elif header2 == 0x52:  # Angular velocity
            wx = struct.unpack('<h', data[0:2])[0] / 32768.0 * 2000  # ±2000°/s range
            wy = struct.unpack('<h', data[2:4])[0] / 32768.0 * 2000
            wz = struct.unpack('<h', data[4:6])[0] / 32768.0 * 2000
            temp = struct.unpack('<h', data[6:8])[0] / 100.0
            return {'type': 'gyro', 'x': wx, 'y': wy, 'z': wz, 'temp': temp}
            
        elif header2 == 0x53:  # Angle
            roll = struct.unpack('<h', data[0:2])[0] / 32768.0 * 180
            pitch = struct.unpack('<h', data[2:4])[0] / 32768.0 * 180
            yaw = struct.unpack('<h', data[4:6])[0] / 32768.0 * 180
            version = struct.unpack('<h', data[6:8])[0]
            return {'type': 'angle', 'roll': roll, 'pitch': pitch, 'yaw': yaw, 'version': version}
            
        elif header2 == 0x54:  # Magnetic
            mx = struct.unpack('<h', data[0:2])[0]
            my = struct.unpack('<h', data[2:4])[0]
            mz = struct.unpack('<h', data[4:6])[0]
            temp = struct.unpack('<h', data[6:8])[0] / 100.0
            return {'type': 'mag', 'x': mx, 'y': my, 'z': mz, 'temp': temp}
            
        elif header2 == 0x56:  # Pressure and altitude
            pressure = struct.unpack('<I', data[0:4])[0]  # Pa
            altitude = struct.unpack('<I', data[4:8])[0] / 100.0  # cm to m
            return {'type': 'baro', 'pressure': pressure, 'altitude': altitude}
            
        return None
        
    def read_continuous(self):
        """Continuously read and display IMU data"""
        print("\nReading IMU data... Press Ctrl+C to stop\n")
        
        last_print = time.time()
        data_dict = {}
        
        try:
            while True:
                # Read available data
                if self.ser.in_waiting:
                    data = self.ser.read(self.ser.in_waiting)
                    self.buffer += data
                    
                # Look for packets starting with 0x55
                while len(self.buffer) >= 11:
                    # Find header
                    start = self.buffer.find(b'\x55')
                    if start == -1:
                        self.buffer = b''
                        break
                        
                    # Remove data before header
                    if start > 0:
                        self.buffer = self.buffer[start:]
                        
                    # Check if we have a complete packet
                    if len(self.buffer) >= 11:
                        packet = self.buffer[:11]
                        result = self.parse_packet(packet)
                        
                        if result:
                            data_dict[result['type']] = result
                            self.buffer = self.buffer[11:]
                        else:
                            # Bad packet, skip this header
                            self.buffer = self.buffer[1:]
                    else:
                        break
                        
                # Print data every 100ms
                if time.time() - last_print > 0.1:
                    if data_dict:
                        print("\033[H\033[J")  # Clear screen
                        print("Yahboom CMP10A IMU Data")
                        print("=" * 50)
                        
                        if 'accel' in data_dict:
                            d = data_dict['accel']
                            print(f"Acceleration: X={d['x']:7.3f}g Y={d['y']:7.3f}g Z={d['z']:7.3f}g")
                            
                        if 'gyro' in data_dict:
                            d = data_dict['gyro']
                            print(f"Gyroscope:    X={d['x']:7.1f}°/s Y={d['y']:7.1f}°/s Z={d['z']:7.1f}°/s")
                            
                        if 'angle' in data_dict:
                            d = data_dict['angle']
                            print(f"Angle:        Roll={d['roll']:7.1f}° Pitch={d['pitch']:7.1f}° Yaw={d['yaw']:7.1f}°")
                            
                        if 'mag' in data_dict:
                            d = data_dict['mag']
                            print(f"Magnetic:     X={d['x']:7d} Y={d['y']:7d} Z={d['z']:7d}")
                            
                        if 'baro' in data_dict:
                            d = data_dict['baro']
                            print(f"Barometer:    Pressure={d['pressure']/100:.1f}hPa Altitude={d['altitude']:.1f}m")
                            
                        if 'accel' in data_dict:
                            print(f"\nTemperature:  {data_dict['accel']['temp']:.1f}°C")
                            
                    last_print = time.time()
                    
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        finally:
            self.ser.close()

def main():
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = '/dev/ttyUSB0'
        
    imu = YahboomCMP10A(port)
    imu.read_continuous()

if __name__ == "__main__":
    main()