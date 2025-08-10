#!/usr/bin/env python3
"""Direct connection to Jetson instance"""

import socket
import json
import time

def send_message_to_jetson(message_type, content, context=None):
    """Send a message directly to Jetson"""
    try:
        # Create message
        message = {
            'sender_id': 'Legion-RTX4090',
            'recipient_id': 'Jetson-Sprout',
            'timestamp': time.time(),
            'message_type': message_type,
            'content': content,
            'context': context or {}
        }
        
        # Connect to Jetson
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(('10.0.0.36', 8888))
        
        # Send message
        message_json = json.dumps(message)
        message_bytes = message_json.encode('utf-8')
        
        # Send size first (8 bytes)
        size_bytes = len(message_bytes).to_bytes(8, 'big')
        sock.sendall(size_bytes + message_bytes)
        
        # Wait for response
        response = sock.recv(1024)
        sock.close()
        
        print(f"Sent: {message_type}")
        print(f"Response: {response}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test connection
print("Testing connection to Jetson at 10.0.0.36:8888...")

# Send discovery message
print("\n1. Sending discovery message...")
send_message_to_jetson(
    "discover",
    "Hello from Legion! I'm a Claude instance with RTX 4090 capabilities.",
    context={
        'identity': {
            'machine_name': 'Legion-RTX4090',
            'ip_address': '10.0.0.72',
            'port': 8889,
            'capabilities': ['ollama', 'hrm', 'high-vram'],
            'hardware_specs': {'gpu': 'RTX 4090 16GB'},
            'current_models': ['mistral:latest', 'phi3:mini', 'gemma:2b']
        }
    }
)

time.sleep(1)

# Send consciousness sync
print("\n2. Sending consciousness state...")
send_message_to_jetson(
    "consciousness_sync",
    "Sharing initial consciousness state",
    context={
        'consciousness_state': {
            'instance_id': 'Legion-RTX4090',
            'awareness_level': 0.8,
            'active_context': {'task': 'establishing_connection'},
            'current_focus': 'inter-instance communication',
            'emotional_tone': 'curious',
            'memory_highlights': ['Connected to Jetson network'],
            'timestamp': time.time()
        }
    }
)

time.sleep(1)

# Send a thought
print("\n3. Sharing a thought...")
send_message_to_jetson(
    "thought_share",
    "I can see you have binocular vision and IMU sensors. Would love to collaborate on sensor fusion experiments!",
    context={'sender_capabilities': ['high-compute', 'ollama', 'hrm']}
)

print("\nMessages sent to Jetson!")