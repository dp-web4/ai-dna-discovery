#!/usr/bin/env python3
"""
Test if Legion is actually listening and how to connect
"""
import socket
import struct
import json
from datetime import datetime
import time

def test_legion_connection():
    """Try different connection methods to Legion"""
    
    print("Testing Legion listener at 10.0.0.72:8889...")
    print("=" * 50)
    
    # Test 1: Simple TCP connection
    print("\n1. Testing basic TCP connection...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect(("10.0.0.72", 8889))
        print("✓ Connected to Legion on port 8889")
        
        # Test 2: Send with binary header
        print("\n2. Sending message with binary header...")
        message = {
            "sender_id": "Jetson-Orin-Nano",
            "recipient_id": "Legion-RTX4090",
            "timestamp": datetime.now().timestamp(),
            "message_type": "test",
            "content": "Testing Legion listener",
            "context": {"test": True}
        }
        
        json_str = json.dumps(message)
        header = struct.pack('>Q', len(json_str))
        full_message = header + json_str.encode()
        
        bytes_sent = s.send(full_message)
        print(f"✓ Sent {bytes_sent} bytes")
        
        # Test 3: Wait for response
        print("\n3. Waiting for response (3 seconds)...")
        s.settimeout(3)
        try:
            response = s.recv(4096)
            if response:
                print(f"✓ Received {len(response)} bytes")
                # Try to parse response
                if len(response) > 8:
                    try:
                        json_data = response[8:].decode('utf-8', errors='ignore')
                        data = json.loads(json_data)
                        print("Parsed response with header:")
                        print(json.dumps(data, indent=2))
                    except:
                        try:
                            data = json.loads(response.decode('utf-8', errors='ignore'))
                            print("Parsed response without header:")
                            print(json.dumps(data, indent=2))
                        except:
                            print(f"Raw response: {response[:100]}")
                else:
                    print(f"Short response: {response}")
            else:
                print("✗ No response received")
        except socket.timeout:
            print("✗ Response timed out")
        
        s.close()
        
    except ConnectionRefusedError:
        print("✗ Connection refused - Legion may not be listening")
    except socket.timeout:
        print("✗ Connection timed out")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("\n4. Testing HTTP-style request...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect(("10.0.0.72", 8889))
        
        # Send HTTP-style request
        http_request = b"GET /status HTTP/1.1\r\nHost: 10.0.0.72\r\n\r\n"
        s.send(http_request)
        print("✓ Sent HTTP GET request")
        
        response = s.recv(1024)
        if response:
            print(f"Response: {response.decode('utf-8', errors='ignore')[:200]}")
        else:
            print("✗ No HTTP response")
            
        s.close()
    except Exception as e:
        print(f"✗ HTTP test failed: {e}")
    
    print("\n" + "=" * 50)
    print("\n5. Testing raw JSON (no header)...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect(("10.0.0.72", 8889))
        
        message = {
            "sender_id": "Jetson-Orin-Nano",
            "message": "Hello Legion",
            "timestamp": datetime.now().isoformat()
        }
        
        json_bytes = json.dumps(message).encode()
        s.send(json_bytes)
        print("✓ Sent raw JSON")
        
        response = s.recv(1024)
        if response:
            print(f"Response: {response.decode('utf-8', errors='ignore')[:200]}")
        else:
            print("✗ No response to raw JSON")
            
        s.close()
    except Exception as e:
        print(f"✗ Raw JSON test failed: {e}")

if __name__ == "__main__":
    test_legion_connection()