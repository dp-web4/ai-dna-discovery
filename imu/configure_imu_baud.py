#!/usr/bin/env python3
"""
Configure Yahboom CMP10A IMU baud rate
Protocol: FF AA <Register> <Value> <Checksum>
"""
import serial
import time
import sys

def calculate_checksum(data):
    """Calculate checksum for command"""
    return sum(data) & 0xFF

def send_command(ser, register, value):
    """Send configuration command to IMU"""
    cmd = [0xFF, 0xAA, register, value]
    checksum = calculate_checksum(cmd)
    cmd.append(checksum)
    
    print(f"Sending: {' '.join(f'{b:02X}' for b in cmd)}")
    ser.write(bytes(cmd))
    time.sleep(0.1)

def configure_baud_rate(port='/dev/ttyUSB0', target_baud=115200):
    """Configure IMU baud rate"""
    
    # Baud rate mapping
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
    
    if target_baud not in baud_map:
        print(f"Error: Unsupported baud rate {target_baud}")
        print(f"Supported rates: {list(baud_map.keys())}")
        return False
        
    print(f"Configuring IMU to {target_baud} baud...")
    
    # First try at current baud rate (9600)
    print("\nTrying at 9600 baud (current setting)...")
    try:
        ser = serial.Serial(port, 9600, timeout=0.5)
        time.sleep(0.1)
        
        # Send unlock command first (if needed)
        send_command(ser, 0x69, 0xB5)  # Unlock
        
        # Send baud rate command
        send_command(ser, 0x04, baud_map[target_baud])
        
        # Send save command
        send_command(ser, 0x00, 0x00)  # Save configuration
        
        ser.close()
        print(f"\nConfiguration sent! IMU should now be at {target_baud} baud.")
        print("Testing new baud rate...")
        
        # Test new baud rate
        time.sleep(0.5)
        ser = serial.Serial(port, target_baud, timeout=0.5)
        data = ser.read(100)
        if data and b'\x55' in data:
            print(f"✓ Success! IMU is now communicating at {target_baud} baud")
            return True
        else:
            print("✗ No data at new baud rate")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        if 'ser' in locals():
            ser.close()

def reset_to_default(port='/dev/ttyUSB0'):
    """Reset IMU to factory defaults"""
    print("Resetting IMU to factory defaults...")
    
    # Try common baud rates to find current setting
    for baud in [9600, 115200, 921600]:
        try:
            print(f"\nTrying at {baud} baud...")
            ser = serial.Serial(port, baud, timeout=0.5)
            time.sleep(0.1)
            
            # Check if we get data
            data = ser.read(50)
            if data and b'\x55' in data:
                print(f"Found IMU at {baud} baud")
                
                # Send factory reset command
                send_command(ser, 0x00, 0xFF)  # Factory reset
                
                ser.close()
                print("Factory reset command sent!")
                return True
                
        except Exception as e:
            print(f"  Failed: {e}")
            
    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Configure CMP10A IMU baud rate')
    parser.add_argument('--baud', type=int, default=115200, 
                       help='Target baud rate (default: 115200)')
    parser.add_argument('--reset', action='store_true', 
                       help='Reset to factory defaults')
    parser.add_argument('--port', default='/dev/ttyUSB0',
                       help='Serial port (default: /dev/ttyUSB0)')
    
    args = parser.parse_args()
    
    if args.reset:
        reset_to_default(args.port)
    else:
        configure_baud_rate(args.port, args.baud)