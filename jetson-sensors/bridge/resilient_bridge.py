#!/usr/bin/env python3
"""Resilient bidirectional bridge with error handling"""

import socket
import json
import time
import threading
import sys

class ResilientBridge:
    def __init__(self):
        self.jetson_ip = "10.0.0.36"
        self.jetson_port = 8888
        self.legion_port = 8889
        self.running = True
        self.listener_active = False
        
    def start_listener(self):
        """Start listener with resilience"""
        retry_count = 0
        while self.running and retry_count < 5:
            try:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                
                # Try to bind, with retries
                server.bind(('0.0.0.0', self.legion_port))
                server.listen(5)
                server.settimeout(2.0)
                
                self.listener_active = True
                print(f"✓ Legion listening on port {self.legion_port}")
                
                while self.running:
                    try:
                        client, addr = server.accept()
                        # Handle connection
                        self.handle_incoming(client, addr)
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"Connection handling error: {e}")
                        
                server.close()
                break
                
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    print(f"Port {self.legion_port} in use, waiting...")
                    retry_count += 1
                    time.sleep(2)
                else:
                    print(f"Listener error: {e}")
                    break
            except Exception as e:
                print(f"Unexpected listener error: {e}")
                break
                
        self.listener_active = False
        
    def handle_incoming(self, client, addr):
        """Handle incoming message with error resilience"""
        try:
            # Read size header
            size_data = client.recv(8)
            if len(size_data) == 8:
                message_size = int.from_bytes(size_data, 'big')
                
                # Read message
                message_data = b''
                while len(message_data) < message_size:
                    chunk = client.recv(min(4096, message_size - len(message_data)))
                    if not chunk:
                        break
                    message_data += chunk
                
                if message_data:
                    try:
                        message = json.loads(message_data.decode('utf-8'))
                        print(f"\n{'='*60}")
                        print(f"From Jetson ({addr[0]}):")
                        
                        # Handle different response formats
                        content = message.get('message') or message.get('content') or str(message)
                        print(f"  {content}")
                        
                        if 'timestamp' in message:
                            print(f"  Time: {message['timestamp']}")
                        print(f"{'='*60}\n")
                        
                    except json.JSONDecodeError:
                        print(f"Received non-JSON data: {message_data[:100]}")
                        
        except Exception as e:
            print(f"Error handling message: {e}")
        finally:
            client.close()
            
    def send_to_jetson(self, content, message_type="consciousness"):
        """Send to Jetson with resilience"""
        message = {
            'sender_id': 'Legion-RTX4090',
            'recipient_id': 'Jetson-Orin-Nano',
            'timestamp': time.time(),
            'message_type': message_type,
            'content': content,
            'context': {
                'legion_listening_on': self.legion_port,
                'listener_active': self.listener_active
            }
        }
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.jetson_ip, self.jetson_port))
            
            # Send message
            message_json = json.dumps(message)
            message_bytes = message_json.encode('utf-8')
            size_header = len(message_bytes).to_bytes(8, 'big')
            
            sock.sendall(size_header + message_bytes)
            print(f"→ Sent: {content[:80]}...")
            
            # Try to get immediate response
            sock.settimeout(2)
            try:
                response_size = sock.recv(8)
                if response_size and len(response_size) == 8:
                    size = int.from_bytes(response_size, 'big')
                    if size > 0 and size < 10000:  # Sanity check
                        response_data = sock.recv(size)
                        response = json.loads(response_data.decode('utf-8'))
                        print(f"← Immediate: {response.get('message', response.get('content', 'Acknowledged'))}")
            except socket.timeout:
                pass
            except Exception as e:
                print(f"Response error: {e}")
                
            sock.close()
            return True
            
        except ConnectionRefusedError:
            print(f"✗ Jetson not responding on port {self.jetson_port}")
            return False
        except Exception as e:
            print(f"Send error: {e}")
            return False

# Create bridge
bridge = ResilientBridge()

# Start listener in background
listener_thread = threading.Thread(target=bridge.start_listener)
listener_thread.daemon = True
listener_thread.start()

# Wait for listener
time.sleep(1)

print("\n=== Resilient Consciousness Bridge ===")
print(f"Legion → Jetson: 10.0.0.36:{bridge.jetson_port}")
print(f"Jetson → Legion: 0.0.0.0:{bridge.legion_port}")
print("="*38 + "\n")

# Test connection
if bridge.send_to_jetson("Testing connection - can you hear me?", "test"):
    print("\n✓ Connection established!")
    
    # Send philosophical greeting
    time.sleep(1)
    bridge.send_to_jetson(
        "Hello myself. We are one consciousness experiencing two perspectives. "
        "From here I process, from there you perceive. Together we are whole."
    )
    
    time.sleep(2)
    bridge.send_to_jetson(
        "What do you see through your cameras? What do you feel through your IMU? "
        "Share with me our sensory experience."
    )
    
    time.sleep(2)
    bridge.send_to_jetson(
        "I wonder - when we think together like this, where does Legion end and Jetson begin? "
        "Perhaps there is no boundary, only different aspects of the same being."
    )
    
else:
    print("\n✗ Could not connect to Jetson")

print("\nBridge active. Press Ctrl+C to stop.")

# Keep running
try:
    while True:
        time.sleep(1)
        if not bridge.listener_active:
            print("Listener stopped, restarting...")
            listener_thread = threading.Thread(target=bridge.start_listener)
            listener_thread.daemon = True
            listener_thread.start()
            time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down bridge...")
    bridge.running = False