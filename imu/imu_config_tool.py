#!/usr/bin/env python3
"""
Yahboom CMP10A IMU Configuration Tool
Based on official protocol documentation
"""
import serial
import time
import sys

class IMUConfig:
    def __init__(self, port='/dev/ttyUSB0'):
        self.port = port
        self.current_baud = 9600  # Default
        
        # Protocol constants
        self.HEADER = [0xFF, 0xAA]
        
        # Register addresses
        self.REG_SAVE = 0x00
        self.REG_BAUD = 0x04
        self.REG_KEY = 0x69
        
        # Baud rate values
        self.BAUD_VALUES = {
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
        
    def send_command(self, register, HL, LL):
        """Send command to IMU"""
        # Build command: FF AA REG HL LL
        cmd = self.HEADER + [register, HL, LL]
        
        # Send via serial
        try:
            with serial.Serial(self.port, self.current_baud, timeout=0.5) as ser:
                print(f"Sending: {' '.join(f'{b:02X}' for b in cmd)}")
                ser.write(bytes(cmd))
                time.sleep(0.1)
                
                # Read any response
                response = ser.read(20)
                if response:
                    print(f"Response: {' '.join(f'{b:02X}' for b in response)}")
                return True
        except Exception as e:
            print(f"Error: {e}")
            return False
            
    def unlock(self):
        """Unlock configuration"""
        print("\n1. Unlocking configuration...")
        return self.send_command(self.REG_KEY, 0x88, 0xB5)
        
    def change_baud(self, new_baud):
        """Change baud rate"""
        if new_baud not in self.BAUD_VALUES:
            print(f"Error: Unsupported baud rate {new_baud}")
            print(f"Supported: {list(self.BAUD_VALUES.keys())}")
            return False
            
        print(f"\n2. Changing baud rate to {new_baud}...")
        value = self.BAUD_VALUES[new_baud]
        return self.send_command(self.REG_BAUD, value, 0x00)
        
    def save_config(self):
        """Save configuration and reboot"""
        print("\n3. Saving configuration (will reboot IMU)...")
        return self.send_command(self.REG_SAVE, 0xFF, 0x00)
        
    def configure(self, target_baud):
        """Complete configuration process"""
        print(f"=== Configuring IMU from {self.current_baud} to {target_baud} baud ===")
        
        # Step 1: Unlock
        if not self.unlock():
            print("Failed to unlock!")
            return False
            
        time.sleep(0.5)
        
        # Step 2: Change baud
        if not self.change_baud(target_baud):
            print("Failed to change baud!")
            return False
            
        time.sleep(0.5)
        
        # Step 3: Save
        if not self.save_config():
            print("Failed to save!")
            return False
            
        print("\n4. Waiting for IMU to reboot...")
        time.sleep(2)
        
        # Test new baud rate
        print(f"\n5. Testing communication at {target_baud} baud...")
        try:
            with serial.Serial(self.port, target_baud, timeout=0.5) as ser:
                # Wait for data
                data = ser.read(100)
                if data and b'\x55' in data:
                    print(f"✓ Success! IMU is now at {target_baud} baud")
                    print(f"  Received {len(data)} bytes")
                    return True
                else:
                    print("✗ No data received at new baud rate")
                    return False
        except Exception as e:
            print(f"✗ Failed to connect at {target_baud}: {e}")
            return False
            
    def detect_current_baud(self):
        """Detect current baud rate"""
        print("Detecting current baud rate...")
        
        for baud in [9600, 115200, 921600, 230400, 57600, 38400, 19200, 4800]:
            try:
                with serial.Serial(self.port, baud, timeout=0.2) as ser:
                    data = ser.read(50)
                    if data and b'\x55' in data:
                        print(f"✓ Found IMU at {baud} baud")
                        self.current_baud = baud
                        return baud
            except:
                pass
                
        print("✗ Could not detect IMU")
        return None
        
    def reset_to_9600(self):
        """Try to reset to 9600 from any baud rate"""
        print("Attempting to reset to 9600 baud...")
        
        # Try each possible baud rate
        for baud in [115200, 921600, 230400, 57600, 38400, 19200, 9600, 4800]:
            self.current_baud = baud
            print(f"\nTrying at {baud} baud...")
            
            if self.unlock():
                time.sleep(0.5)
                if self.change_baud(9600):
                    time.sleep(0.5)
                    if self.save_config():
                        print("Reset command sent!")
                        return True
                        
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Configure CMP10A IMU baud rate')
    parser.add_argument('baud', type=int, nargs='?', default=115200,
                       help='Target baud rate (default: 115200)')
    parser.add_argument('--port', default='/dev/ttyUSB0',
                       help='Serial port')
    parser.add_argument('--detect', action='store_true',
                       help='Just detect current baud rate')
    parser.add_argument('--reset', action='store_true',
                       help='Reset to 9600 baud')
    
    args = parser.parse_args()
    
    config = IMUConfig(args.port)
    
    if args.detect:
        config.detect_current_baud()
    elif args.reset:
        config.reset_to_9600()
    else:
        # Detect current baud
        current = config.detect_current_baud()
        if current:
            if current == args.baud:
                print(f"Already at {args.baud} baud!")
            else:
                config.configure(args.baud)
        else:
            print("Could not detect IMU. Is it connected?")

if __name__ == "__main__":
    main()