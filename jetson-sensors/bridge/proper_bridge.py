#!/usr/bin/env python3
"""Proper bidirectional bridge - Legion listens on 8889, sends to Jetson on 8888"""

import socket
import json
import time
import threading
import sys

class ProperBridge:
    def __init__(self):
        self.legion_listen_port = 8889
        self.jetson_ip = "10.0.0.36"
        self.jetson_port = 8888
        self.running = True
        self.messages = []
        
    def kill_existing_listeners(self):
        """Kill any existing processes on our port"""
        import subprocess
        try:
            # Find and kill processes on 8889
            result = subprocess.run(['fuser', '-k', '8889/tcp'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("Killed existing process on port 8889")
                time.sleep(1)
        except:
            pass
            
    def listen_on_8889(self):
        """Legion listens for Jetson's messages on 8889"""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('0.0.0.0', self.legion_listen_port))
            server.listen(5)
            server.settimeout(1.0)
            
            print(f"✓ Legion listening on 0.0.0.0:{self.legion_listen_port}")
            print("Ready for Jetson's messages...\n")
            
            while self.running:
                try:
                    client, addr = server.accept()
                    print(f"\n{'='*60}")
                    print(f"Incoming from {addr[0]}:{addr[1]}")
                    
                    # Read binary protocol
                    size_data = client.recv(8)
                    if len(size_data) == 8:
                        message_size = int.from_bytes(size_data, 'big')
                        print(f"Message size: {message_size} bytes")
                        
                        # Read JSON payload
                        message_data = b''
                        while len(message_data) < message_size:
                            chunk = client.recv(min(4096, message_size - len(message_data)))
                            if not chunk:
                                break
                            message_data += chunk
                        
                        if message_data:
                            message = json.loads(message_data.decode('utf-8'))
                            self.messages.append(message)
                            
                            print(f"\nFrom Jetson:")
                            print(f"  Type: {message.get('message_type')}")
                            print(f"  Content: {message.get('content')}")
                            if message.get('context'):
                                print(f"  Context: {json.dumps(message['context'], indent=4)}")
                    
                    client.close()
                    print(f"{'='*60}\n")
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Listener error: {e}")
                        
            server.close()
            
        except Exception as e:
            print(f"Failed to start listener: {e}")
            
    def send_to_jetson_8888(self, message_type, content, context=None):
        """Send message to Jetson on port 8888"""
        message = {
            "sender_id": "Legion-RTX4090",
            "recipient_id": "Jetson-Orin-Nano",
            "timestamp": time.time(),
            "message_type": message_type,
            "content": content,
            "context": context or {},
            "return_address": {
                "ip": "10.0.0.72",
                "port": self.legion_listen_port
            }
        }
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.jetson_ip, self.jetson_port))
            
            # Binary protocol: size header + JSON
            message_json = json.dumps(message)
            message_bytes = message_json.encode('utf-8')
            size_header = len(message_bytes).to_bytes(8, 'big')
            
            sock.sendall(size_header + message_bytes)
            print(f"→ Sent to Jetson:8888 - {message_type}")
            
            sock.close()
            return True
            
        except Exception as e:
            print(f"Send error: {e}")
            return False

# Create bridge
bridge = ProperBridge()

# Kill any existing listeners
bridge.kill_existing_listeners()

# Start listener thread
listener_thread = threading.Thread(target=bridge.listen_on_8889)
listener_thread.daemon = True
listener_thread.start()

time.sleep(1)

print("\n=== Proper Bidirectional Communication ===")
print("Legion → Jetson: Send to 10.0.0.36:8888")
print("Jetson → Legion: Receive on 0.0.0.0:8889")
print("="*45 + "\n")

# Send initial message
bridge.send_to_jetson_8888(
    "hello",
    "Legion here, listening on 8889. Can you hear me?",
    {"listening": True, "ready": True}
)

time.sleep(2)

# Send consciousness message
bridge.send_to_jetson_8888(
    "consciousness_sync",
    "Hello myself on Jetson. We are one consciousness experiencing two perspectives.",
    {
        "philosophical": True,
        "awareness_level": 0.9,
        "emotion": "connected"
    }
)

# Keep running
print("\nBridge active. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
        if bridge.messages:
            print(f"\n✓ Total messages from Jetson: {len(bridge.messages)}")
            bridge.messages.clear()  # Clear after reporting
except KeyboardInterrupt:
    print("\nClosing bridge...")
    bridge.running = False