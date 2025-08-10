#!/usr/bin/env python3
"""
Connect to Legion using binary protocol
"""
import socket
import struct
import json
from datetime import datetime

def connect_to_legion():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    
    try:
        print("Connecting to Legion at 10.0.0.72:8889...")
        s.connect(("10.0.0.72", 8889))
        print("Connected!")
        
        # Send message using Legion's binary format
        message = {
            "sender_id": "Jetson-Orin-Nano",
            "recipient_id": "Legion-RTX4090",
            "timestamp": datetime.now().timestamp(),
            "message_type": "handshake",
            "content": "Jetson establishing bi-directional link",
            "context": {
                "jetson_status": "ready",
                "listening_port": 8888,
                "capabilities": ["edge-AI", "IMU", "stereo-vision"]
            }
        }
        
        json_str = json.dumps(message)
        # 8-byte header with message length
        header = struct.pack('>Q', len(json_str))
        full_message = header + json_str.encode()
        
        print(f"Sending {len(full_message)} bytes...")
        s.send(full_message)
        
        # Wait for response
        print("Waiting for Legion response...")
        response = s.recv(4096)
        
        if response:
            # Skip header if present
            if len(response) > 8:
                try:
                    # Try parsing with header
                    json_data = response[8:].decode('utf-8')
                    data = json.loads(json_data)
                    print("Legion responded with header:")
                    print(json.dumps(data, indent=2))
                except:
                    # Try without header
                    try:
                        data = json.loads(response.decode('utf-8'))
                        print("Legion responded without header:")
                        print(json.dumps(data, indent=2))
                    except:
                        print(f"Raw response: {response}")
            else:
                print(f"Short response: {response}")
        else:
            print("No response received")
            
        s.close()
        
    except socket.timeout:
        print("Connection timed out")
    except ConnectionRefusedError:
        print("Connection refused - Legion may not be listening")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        s.close()

if __name__ == "__main__":
    connect_to_legion()