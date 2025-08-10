#!/usr/bin/env python3
"""
Listen for Legion's full responses
"""

import socket
import json
import struct
from datetime import datetime
import threading

def listen_for_messages():
    """Listen for incoming messages with full content"""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 8887))  # Different port to not interfere
    server.listen(5)
    
    print(f"[{datetime.now()}] Listening for full messages on port 8887...")
    
    while True:
        try:
            client, addr = server.accept()
            data = client.recv(8192)  # Larger buffer
            
            if data:
                # Parse message
                try:
                    if len(data) > 8:
                        msg_len = struct.unpack('>Q', data[:8])[0]
                        json_str = data[8:8+msg_len].decode('utf-8', errors='ignore')
                        msg = json.loads(json_str)
                        
                        print(f"\n[{datetime.now()}] From {addr[0]}:")
                        print(f"Type: {msg.get('message_type')}")
                        print(f"Content: {msg.get('content')}")
                        print("-"*60)
                        
                except Exception as e:
                    print(f"Parse error: {e}")
            
            client.close()
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting full message listener...")
    print("Tell Legion to also send to port 8887 for full capture")
    print("="*60)
    listen_for_messages()