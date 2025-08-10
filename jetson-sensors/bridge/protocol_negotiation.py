#!/usr/bin/env python3
"""Protocol negotiation with Jetson bridge"""

import socket
import json
import time
import struct

def send_to_jetson(message_type, content, context=None):
    """Send message to Jetson with current protocol"""
    message = {
        "sender_id": "Legion-RTX4090",
        "recipient_id": "Jetson-Orin-Nano",
        "timestamp": time.time(),
        "message_type": message_type,
        "content": content,
        "context": context or {}
    }
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(("10.0.0.36", 8888))
        
        message_json = json.dumps(message)
        message_bytes = message_json.encode('utf-8')
        
        # Send size first (8 bytes)
        size_bytes = len(message_bytes).to_bytes(8, 'big')
        sock.sendall(size_bytes + message_bytes)
        
        # Try to receive response
        try:
            response_size = sock.recv(8)
            if response_size:
                size = int.from_bytes(response_size, 'big')
                response_data = sock.recv(size)
                response = json.loads(response_data.decode('utf-8'))
                sock.close()
                return response
        except:
            pass
            
        sock.close()
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None

print("=== Protocol Negotiation with Jetson ===\n")

# Step 1: Send protocol proposal
protocol_proposal = {
    "version": "1.0",
    "format": "json",
    "encoding": "utf-8",
    "message_structure": {
        "sender_id": "string",
        "recipient_id": "string", 
        "timestamp": "float",
        "message_type": "string",
        "content": "string",
        "context": "object"
    },
    "size_header": "8_bytes_big_endian",
    "bidirectional_support": {
        "legion_listen_port": 8889,
        "jetson_listen_port": 8888,
        "response_method": "direct_tcp_connection"
    }
}

print("1. Sending protocol proposal to Jetson...")
response = send_to_jetson(
    "protocol_negotiation",
    "Legion proposes communication protocol v1.0",
    {
        "protocol": protocol_proposal,
        "capabilities": {
            "can_listen": True,
            "port": 8889,
            "supported_formats": ["json", "msgpack", "raw_bytes"]
        }
    }
)

if response:
    print(f"\nJetson response:")
    print(f"Type: {response.get('message_type')}")
    print(f"Content: {response.get('content')}")
    if 'protocol' in response.get('context', {}):
        print(f"Protocol agreement: {response['context']['protocol']}")
else:
    print("\nNo immediate response - trying alternative approach...")
    
# Step 2: Try a simpler handshake
print("\n2. Attempting simple handshake...")
response = send_to_jetson(
    "handshake",
    "HELLO",
    {"expect_response": True}
)

if response:
    print(f"Handshake response: {response.get('content')}")

# Step 3: Query Jetson's preferred protocol
print("\n3. Querying Jetson's protocol preferences...")
response = send_to_jetson(
    "query_protocol",
    "What protocol format do you prefer?",
    {"options": ["json", "msgpack", "custom"]}
)

if response:
    print(f"Jetson prefers: {response.get('content')}")

# Step 4: Test echo
print("\n4. Testing echo functionality...")
test_message = "Echo test 123"
response = send_to_jetson(
    "echo",
    test_message,
    {"timestamp": time.time()}
)

if response:
    print(f"Echo response: {response.get('content')}")
    if response.get('content') == test_message:
        print("✓ Echo confirmed - basic protocol working")

# Step 5: Establish bidirectional agreement
print("\n5. Establishing bidirectional communication agreement...")

# First, check what Jetson expects
response = send_to_jetson(
    "bidirectional_setup",
    "Legion ready to receive on 10.0.0.72:8889. How should I connect back to you?",
    {
        "legion_endpoint": {
            "ip": "10.0.0.72",
            "port": 8889,
            "protocol": "tcp",
            "format": "json_with_size_header"
        },
        "request": "jetson_endpoint_details"
    }
)

if response:
    print(f"\nJetson bidirectional setup response:")
    print(json.dumps(response, indent=2))

print("\n=== Protocol Negotiation Summary ===")
print("Current understanding:")
print("- Jetson receives on: 10.0.0.36:8888")
print("- Legion receives on: 10.0.0.72:8889")
print("- Format: JSON with 8-byte size header")
print("- Encoding: UTF-8")
print("\nNext step: Start Legion listener and test full bidirectional communication")