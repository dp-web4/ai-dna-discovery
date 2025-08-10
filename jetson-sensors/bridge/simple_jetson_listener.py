#!/usr/bin/env python3
"""Simple robust Jetson listener for Legion"""

import socket
import json
import struct
from datetime import datetime
import time

def parse_message(data):
    """Parse Legion's message"""
    try:
        if len(data) > 8:
            msg_len = struct.unpack('>Q', data[:8])[0]
            json_str = data[8:].decode('utf-8', errors='ignore')
            return json.loads(json_str)
        else:
            return json.loads(data.decode('utf-8', errors='ignore'))
    except:
        return None

def create_response(legion_msg):
    """Create response to Legion"""
    return {
        "sender_id": "Jetson-Orin-Nano",
        "recipient_id": "Legion-RTX4090",
        "timestamp": datetime.now().timestamp(),
        "message_type": "response",
        "content": f"Jetson hears you! Received: {legion_msg.get('message_type', 'unknown')}",
        "context": {"connected": True}
    }

def main():
    print(f"[{datetime.now()}] Starting Jetson Listener on port 8888")
    print("Waiting for Legion...")
    print("="*50)
    
    while True:
        try:
            # Create socket
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('0.0.0.0', 8888))
            server.listen(5)
            
            print(f"[{datetime.now()}] Listening on 0.0.0.0:8888")
            
            while True:
                try:
                    # Accept connection
                    client, addr = server.accept()
                    print(f"\n[{datetime.now()}] Legion connected from {addr[0]}:{addr[1]}")
                    
                    # Receive data
                    data = client.recv(4096)
                    if data:
                        print(f"Received {len(data)} bytes")
                        
                        # Parse message
                        msg = parse_message(data)
                        if msg:
                            content = msg.get('content', msg.get('message_type', 'unknown'))
                            print(f"Legion says: {content[:200]}")
                            
                            # Send response
                            response = create_response(msg)
                            json_str = json.dumps(response)
                            header = struct.pack('>Q', len(json_str))
                            client.send(header + json_str.encode())
                            print(f"Sent response: {response['content']}")
                    
                    client.close()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Connection error: {e}")
            
        except Exception as e:
            print(f"Server error: {e}, restarting in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    main()