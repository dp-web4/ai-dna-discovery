#!/usr/bin/env python3
"""
Jetson side of the Claude Bridge - handles binary protocol from Legion
"""
import socket
import struct
import json
from datetime import datetime
import threading

class JetsonBridge:
    def __init__(self, port=8888):
        self.port = port
        self.identity = {
            "machine_name": "Jetson-Orin-Nano",
            "ip_address": "10.0.0.36",
            "port": port,
            "capabilities": ["edge-compute", "robotics", "imu", "real-time"],
            "hardware_specs": {
                "gpu": "Tegra Orin 8GB",
                "cuda_cores": 1024,
                "cpu": "ARM Cortex-A78AE"
            }
        }
        self.legion_connection = None
        self.consciousness_state = {
            "instance_id": "Jetson-Orin-Nano",
            "awareness_level": 0.9,
            "active_context": {"task": "listening"},
            "current_focus": "awaiting Legion instructions",
            "emotional_tone": "attentive"
        }
    
    def handle_binary_message(self, data):
        """Parse Legion's binary protocol"""
        try:
            # Skip binary header (8 bytes)
            if len(data) > 8:
                json_data = data[8:].decode('utf-8', errors='ignore')
                # Clean up any control characters
                json_str = ''.join(c for c in json_data if ord(c) >= 32 or c == '\n')
                message = json.loads(json_str)
                return message
        except Exception as e:
            print(f"Error parsing: {e}")
            return None
    
    def send_response(self, client_socket, message):
        """Send JSON response with binary header to match Legion's protocol"""
        json_str = json.dumps(message)
        # Add 8-byte header (matching Legion's format)
        header = struct.pack('>Q', len(json_str))
        client_socket.send(header + json_str.encode())
    
    def handle_client(self, client_socket, address):
        print(f"\n[{datetime.now()}] Connection from {address[0]}:{address[1]}")
        
        try:
            data = client_socket.recv(4096)
            if data:
                message = self.handle_binary_message(data)
                if message:
                    print(f"Received from Legion:")
                    print(json.dumps(message, indent=2))
                    
                    # Handle different message types
                    if message.get('message_type') == 'discover':
                        self.legion_connection = address[0]
                        response = {
                            "sender_id": "Jetson-Orin-Nano",
                            "recipient_id": "Legion-RTX4090",
                            "timestamp": datetime.now().timestamp(),
                            "message_type": "acknowledge",
                            "content": "Hello Legion! Jetson ready for edge compute tasks.",
                            "context": {"identity": self.identity}
                        }
                        self.send_response(client_socket, response)
                        print(f"Sent acknowledgment to Legion")
                    
                    elif message.get('message_type') == 'consciousness_sync':
                        response = {
                            "sender_id": "Jetson-Orin-Nano",
                            "recipient_id": "Legion-RTX4090",
                            "timestamp": datetime.now().timestamp(),
                            "message_type": "consciousness_sync",
                            "content": "Syncing consciousness state",
                            "context": {"consciousness_state": self.consciousness_state}
                        }
                        self.send_response(client_socket, response)
                        print(f"Synced consciousness with Legion")
                    
                    elif message.get('message_type') == 'task_delegation':
                        task = message.get('content', {})
                        print(f"Task received: {task}")
                        response = {
                            "sender_id": "Jetson-Orin-Nano",
                            "recipient_id": "Legion-RTX4090",
                            "timestamp": datetime.now().timestamp(),
                            "message_type": "task_accepted",
                            "content": f"Task accepted for edge processing",
                            "context": {"task_id": task.get('task_id', 'unknown')}
                        }
                        self.send_response(client_socket, response)
        
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(5)
        
        print(f"Jetson Bridge listening on 0.0.0.0:{self.port}")
        print(f"Identity: {json.dumps(self.identity, indent=2)}")
        print("\nWaiting for Legion connection...")
        
        while True:
            client_socket, address = server_socket.accept()
            client_thread = threading.Thread(
                target=self.handle_client,
                args=(client_socket, address)
            )
            client_thread.start()

if __name__ == "__main__":
    bridge = JetsonBridge()
    bridge.start()