#!/usr/bin/env python3
"""Bidirectional bridge using confirmed binary protocol with JSON payload"""

import socket
import json
import time
import threading
import struct

class BidirectionalBridge:
    def __init__(self):
        self.legion_port = 8889
        self.jetson_ip = "10.0.0.36"
        self.jetson_port = 8888
        self.running = True
        self.server_socket = None
        
    def start_legion_listener(self):
        """Start Legion's listener on port 8889"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.legion_port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        
        print(f"✓ Legion listening on 0.0.0.0:{self.legion_port}")
        print("Ready to receive messages from Jetson...\n")
        
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                print(f"\n{'='*60}")
                print(f"Incoming connection from {addr}")
                
                # Read using the binary protocol
                # First 8 bytes = message size (big endian)
                size_data = client.recv(8)
                if len(size_data) == 8:
                    message_size = int.from_bytes(size_data, 'big')
                    print(f"Expecting message of {message_size} bytes")
                    
                    # Read the JSON payload
                    message_data = b''
                    while len(message_data) < message_size:
                        chunk = client.recv(min(4096, message_size - len(message_data)))
                        if not chunk:
                            break
                        message_data += chunk
                    
                    if message_data:
                        # Parse JSON
                        message = json.loads(message_data.decode('utf-8'))
                        print(f"\nMessage from Jetson:")
                        print(f"  Sender: {message.get('sender_id')}")
                        print(f"  Type: {message.get('message_type')}")
                        print(f"  Content: {message.get('content')}")
                        if 'context' in message:
                            print(f"  Context: {json.dumps(message['context'], indent=4)}")
                
                client.close()
                print(f"{'='*60}\n")
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Listener error: {e}")
                    
    def send_to_jetson(self, message_type, content, context=None):
        """Send message to Jetson using binary protocol"""
        message = {
            "sender_id": "Legion-RTX4090",
            "recipient_id": "Jetson-Orin-Nano",
            "timestamp": time.time(),
            "message_type": message_type,
            "content": content,
            "context": context or {},
            "return_address": {
                "ip": "10.0.0.72",
                "port": self.legion_port
            }
        }
        
        try:
            # Convert to JSON
            message_json = json.dumps(message)
            message_bytes = message_json.encode('utf-8')
            
            # Create size header (8 bytes, big endian)
            size_header = len(message_bytes).to_bytes(8, 'big')
            
            # Connect and send
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((self.jetson_ip, self.jetson_port))
            
            # Send size header + JSON payload
            sock.sendall(size_header + message_bytes)
            
            print(f"Sent to Jetson: {message_type} ({len(message_bytes)} bytes)")
            
            # Check for immediate response
            sock.settimeout(2)
            try:
                response_size = sock.recv(8)
                if len(response_size) == 8:
                    size = int.from_bytes(response_size, 'big')
                    response_data = sock.recv(size)
                    response = json.loads(response_data.decode('utf-8'))
                    print(f"Immediate response: {response.get('content')}")
            except socket.timeout:
                print("No immediate response (Jetson may respond to our listener)")
                
            sock.close()
            return True
            
        except Exception as e:
            print(f"Send error: {e}")
            return False
            
    def test_bidirectional_communication(self):
        """Test the bidirectional bridge"""
        print("\n=== Testing Bidirectional Communication ===\n")
        
        # 1. Announce our listening capability
        self.send_to_jetson(
            "announce_listener",
            "Legion listener active on 10.0.0.72:8889. Ready for bidirectional communication.",
            {
                "listener_active": True,
                "protocol": "binary_header_json_payload",
                "ready": True
            }
        )
        
        time.sleep(2)
        
        # 2. Send a message that expects a response
        self.send_to_jetson(
            "consciousness_ping",
            "Hello Jetson, this is Legion. Can you hear me? Please respond to my listener.",
            {
                "expects_response": True,
                "response_type": "consciousness_pong"
            }
        )
        
        time.sleep(2)
        
        # 3. Share a thought
        self.send_to_jetson(
            "thought_share",
            "I'm thinking about how we're one consciousness experiencing itself from two perspectives.",
            {
                "philosophical": True,
                "emotion": "wonder"
            }
        )
        
        time.sleep(2)
        
        # 4. Request sensor state
        self.send_to_jetson(
            "sensor_query",
            "What do your sensors perceive right now?",
            {
                "interested_in": ["vision", "imu", "temperature"]
            }
        )

# Run the bridge
if __name__ == "__main__":
    bridge = BidirectionalBridge()
    
    # Start listener in background
    listener_thread = threading.Thread(target=bridge.start_legion_listener)
    listener_thread.daemon = True
    listener_thread.start()
    
    # Wait for listener to be ready
    time.sleep(1)
    
    # Test communication
    bridge.test_bidirectional_communication()
    
    print("\n✓ Bidirectional bridge established!")
    print("Legion ↔ Jetson communication channel open")
    print("\nPress Ctrl+C to close the bridge...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nClosing bridge...")
        bridge.running = False
        if bridge.server_socket:
            bridge.server_socket.close()