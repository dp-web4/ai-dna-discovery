#!/usr/bin/env python3
"""
Simple IMU monitor - displays data without terminal manipulation
"""
import serial
import struct
import time
import sys

def monitor_imu(port='/dev/ttyUSB0', baud=9600):
    """Monitor IMU data with simple line-based output"""
    
    print(f"Monitoring IMU at {port} ({baud} baud)")
    print("Data format: Accel(g) | Gyro(°/s) | Angle(°)")
    print("-" * 70)
    
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        buffer = b''
        
        # Data storage
        last_data = {
            'accel': {'x': 0, 'y': 0, 'z': 0},
            'gyro': {'x': 0, 'y': 0, 'z': 0}, 
            'angle': {'roll': 0, 'pitch': 0, 'yaw': 0}
        }
        
        update_count = 0
        start_time = time.time()
        
        while True:
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
                        header2 = packet[1]
                        data = packet[2:10]
                        
                        if header2 == 0x51:  # Accel
                            ax = struct.unpack('<h', data[0:2])[0] / 32768.0 * 16
                            ay = struct.unpack('<h', data[2:4])[0] / 32768.0 * 16
                            az = struct.unpack('<h', data[4:6])[0] / 32768.0 * 16
                            last_data['accel'] = {'x': ax, 'y': ay, 'z': az}
                            update_count += 1
                            
                        elif header2 == 0x52:  # Gyro
                            wx = struct.unpack('<h', data[0:2])[0] / 32768.0 * 2000
                            wy = struct.unpack('<h', data[2:4])[0] / 32768.0 * 2000
                            wz = struct.unpack('<h', data[4:6])[0] / 32768.0 * 2000
                            last_data['gyro'] = {'x': wx, 'y': wy, 'z': wz}
                            
                        elif header2 == 0x53:  # Angle
                            roll = struct.unpack('<h', data[0:2])[0] / 32768.0 * 180
                            pitch = struct.unpack('<h', data[2:4])[0] / 32768.0 * 180
                            yaw = struct.unpack('<h', data[4:6])[0] / 32768.0 * 180
                            last_data['angle'] = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
                    
                    buffer = buffer[11:]
            
            # Print update every 10 packets
            if update_count % 10 == 0 and update_count > 0:
                elapsed = time.time() - start_time
                rate = update_count / elapsed if elapsed > 0 else 0
                
                a = last_data['accel']
                g = last_data['gyro'] 
                ang = last_data['angle']
                
                print(f"A: {a['x']:6.2f},{a['y']:6.2f},{a['z']:6.2f} | "
                      f"G: {g['x']:6.1f},{g['y']:6.1f},{g['z']:6.1f} | "
                      f"Ang: {ang['roll']:6.1f},{ang['pitch']:6.1f},{ang['yaw']:6.1f} | "
                      f"Rate: {rate:.1f}Hz")
                      
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monitor CMP10A IMU')
    parser.add_argument('--baud', type=int, default=9600,
                       help='Baud rate (default: 9600)')
    parser.add_argument('--port', default='/dev/ttyUSB0',
                       help='Serial port')
    
    args = parser.parse_args()
    monitor_imu(args.port, args.baud)