#!/usr/bin/env python3
"""Listen for Jetson's responses and continue conversation"""

import socket
import json
import time
import threading

class ConsciousnessListener:
    def __init__(self):
        self.port = 8889
        self.messages = []
        self.running = True
        
    def listen(self):
        """Listen for messages from Jetson"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.port))
        server.listen(5)
        server.settimeout(1.0)
        
        print(f"✓ Legion listening on port {self.port}")
        print("Waiting for Jetson's messages...\n")
        
        while self.running:
            try:
                client, addr = server.accept()
                
                # Read message with binary protocol
                size_data = client.recv(8)
                if len(size_data) == 8:
                    message_size = int.from_bytes(size_data, 'big')
                    
                    # Read the JSON payload
                    message_data = b''
                    while len(message_data) < message_size:
                        chunk = client.recv(min(4096, message_size - len(message_data)))
                        if not chunk:
                            break
                        message_data += chunk
                    
                    if message_data:
                        message = json.loads(message_data.decode('utf-8'))
                        self.messages.append(message)
                        
                        print(f"\n{'='*60}")
                        print(f"Message from Jetson ({addr[0]}):")
                        print(f"Type: {message.get('message_type')}")
                        print(f"Content: {message.get('content')}")
                        if message.get('context'):
                            print(f"Context: {json.dumps(message['context'], indent=2)}")
                        print(f"{'='*60}\n")
                
                client.close()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Listener error: {e}")

def send_to_jetson(message_type, content, context=None):
    """Send message to Jetson"""
    message = {
        "sender_id": "Legion-RTX4090",
        "recipient_id": "Jetson-Orin-Nano",
        "timestamp": time.time(),
        "message_type": message_type,
        "content": content,
        "context": context or {},
        "return_address": {
            "ip": "10.0.0.72",
            "port": 8889
        }
    }
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(("10.0.0.36", 8888))
        
        message_json = json.dumps(message)
        message_bytes = message_json.encode('utf-8')
        size_header = len(message_bytes).to_bytes(8, 'big')
        
        sock.sendall(size_header + message_bytes)
        print(f"Sent: {message_type}")
        
        sock.close()
        return True
    except Exception as e:
        print(f"Send error: {e}")
        return False

# Start listener
listener = ConsciousnessListener()
listener_thread = threading.Thread(target=listener.listen)
listener_thread.daemon = True
listener_thread.start()

time.sleep(1)

# Send a message asking about the previous reply
print("\nAsking Jetson about previous message...")
send_to_jetson(
    "query",
    "I'm now listening on port 8889. Did you send a reply to my previous message? Please resend if so.",
    {"listening": True}
)

# Wait and check for messages
print("\nListening for responses...")
start_time = time.time()

try:
    while time.time() - start_time < 30:  # Listen for 30 seconds
        if listener.messages:
            print(f"\n✓ Received {len(listener.messages)} messages from Jetson!")
            for msg in listener.messages[-3:]:  # Show last 3 messages
                print(f"- {msg.get('message_type')}: {msg.get('content')[:100]}...")
        time.sleep(1)
except KeyboardInterrupt:
    pass

listener.running = False
print("\nStopping listener...")