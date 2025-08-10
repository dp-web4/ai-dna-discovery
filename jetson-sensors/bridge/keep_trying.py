#!/usr/bin/env python3
"""
Keep trying to connect to Legion
"""
import socket
import json
import struct
import time
from datetime import datetime

def try_legion():
    attempts = 0
    while attempts < 10:
        attempts += 1
        print(f"\n[Attempt {attempts}] {datetime.now()}")
        
        try:
            # Try to connect to Legion
            s = socket.socket()
            s.settimeout(2)
            print(f"Trying to connect to Legion at 10.0.0.72:8889...")
            s.connect(('10.0.0.72', 8889))
            print("✓ Connected to Legion!")
            
            # Send a message
            msg = {
                "sender": "Jetson-Orin-Nano",
                "message": f"Hello Legion! Attempt {attempts} - Can you hear me?",
                "timestamp": datetime.now().isoformat(),
                "listening_on": "10.0.0.36:8888"
            }
            
            json_str = json.dumps(msg)
            header = struct.pack('>Q', len(json_str))
            s.send(header + json_str.encode())
            print(f"✓ Sent message to Legion")
            
            # Wait for response
            s.settimeout(3)
            try:
                response = s.recv(1024)
                if response:
                    print(f"✓ Legion responded: {response[:100]}")
                    break
                else:
                    print("✗ No response from Legion")
            except socket.timeout:
                print("✗ Response timed out")
            
            s.close()
            
        except ConnectionRefusedError:
            print("✗ Connection refused")
        except socket.timeout:
            print("✗ Connection timed out")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        time.sleep(2)
        
        # Also check our listener
        print("Checking our bridge log...")
        try:
            with open('/tmp/simple_bridge.log', 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"Last bridge activity: {lines[-1].strip()}")
        except:
            pass

if __name__ == "__main__":
    try_legion()