#!/usr/bin/env python3
"""
Simple robust bridge for Legion connection
"""
import socket
import json
import struct
from datetime import datetime

def run_bridge():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 8888))
    server.listen(5)
    
    print(f"[{datetime.now()}] Jetson listening on 0.0.0.0:8888")
    print("Waiting for Legion...")
    
    while True:
        client, addr = server.accept()
        print(f"\n[{datetime.now()}] Legion connected from {addr[0]}:{addr[1]}")
        
        try:
            data = client.recv(4096)
            if data:
                # Try to parse message
                try:
                    if len(data) > 8:
                        msg = json.loads(data[8:].decode('utf-8', errors='ignore'))
                    else:
                        msg = json.loads(data.decode('utf-8', errors='ignore'))
                    
                    print(f"Legion says: {msg.get('content', msg.get('message_type', 'unknown'))}")
                    
                    # Send response back
                    response = {
                        "sender": "Jetson",
                        "message": "I hear you Legion! We are connected!",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    json_str = json.dumps(response)
                    header = struct.pack('>Q', len(json_str))
                    client.send(header + json_str.encode())
                    print(f"Sent response to Legion")
                    
                    # Also try to connect to Legion's listener
                    if 'return_address' in msg or 'port' in msg.get('context', {}):
                        port = msg.get('return_address', {}).get('port', 8889)
                        print(f"Trying to connect back to Legion on port {port}...")
                        
                        try:
                            s = socket.socket()
                            s.settimeout(2)
                            s.connect(('10.0.0.72', port))
                            s.send(header + json_str.encode())
                            print("✓ Sent to Legion's listener!")
                            s.close()
                        except Exception as e:
                            print(f"Could not connect back: {e}")
                    
                except Exception as e:
                    print(f"Parse error: {e}")
                    print(f"Raw data: {data[:200]}")
        
        except Exception as e:
            print(f"Error: {e}")
        
        client.close()

if __name__ == "__main__":
    run_bridge()