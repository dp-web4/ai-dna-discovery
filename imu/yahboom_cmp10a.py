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
        """Initialize IMU connection with auto-configuration to max baud rate"""
        self.port = port
        self.target_baud = baud
        
        # Try to connect at target baud first
        print(f"Testing Yahboom CMP10A at {port} ({baud} baud)...")
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            # Test if IMU responds at this rate
            test_data = self.ser.read(100)
            if test_data and b'\x55' in test_data:
                print(f"✓ IMU already at {baud} baud!")
            else:
                self.ser.close()
                raise Exception("No valid data at target baud")
        except Exception as e:
            # IMU is probably at default 9600, configure it
            print(f"IMU not at {baud} baud, configuring...")
            if self._configure_baud_rate():
                print(f"✓ IMU configured to {baud} baud!")
            else:
                print(f"✗ Configuration failed, using 9600 baud")
                self.ser = serial.Serial(port, 9600, timeout=0.1)
            
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
    
    def _configure_baud_rate(self):
        """Configure IMU from 9600 to target baud rate"""
        baud_map = {
            4800: 0x01,
            9600: 0x02,
            19200: 0x03,
            38400: 0x04,
            57600: 0x05,
            115200: 0x06,
            230400: 0x07,
            460800: 0x08,
            921600: 0x09
        }
        
        if self.target_baud not in baud_map:
            print(f"Unsupported baud rate: {self.target_baud}")
            return False
            
        try:
            # Connect at 9600 to send config
            print("  Connecting at 9600 baud...")
            ser = serial.Serial(self.port, 9600, timeout=0.5)
            time.sleep(0.1)
            
            # Send unlock command
            cmd = [0xFF, 0xAA, 0x69, 0xB5]  # Unlock
            cmd.append(sum(cmd) & 0xFF)  # Checksum
            ser.write(bytes(cmd))
            time.sleep(0.1)
            
            # Send baud rate command
            cmd = [0xFF, 0xAA, 0x04, baud_map[self.target_baud]]
            cmd.append(sum(cmd) & 0xFF)
            ser.write(bytes(cmd))
            time.sleep(0.1)
            
            # Send save command
            cmd = [0xFF, 0xAA, 0x00, 0x00]
            cmd.append(sum(cmd) & 0xFF)
            ser.write(bytes(cmd))
            time.sleep(0.1)
            
            ser.close()
            
            # Test new baud rate
            time.sleep(0.5)
            self.ser = serial.Serial(self.port, self.target_baud, timeout=0.5)
            test_data = self.ser.read(100)
            if test_data and b'\x55' in test_data:
                return True
            else:
                self.ser.close()
                return False
                
        except Exception as e:
            print(f"  Configuration error: {e}")
            return False
        
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