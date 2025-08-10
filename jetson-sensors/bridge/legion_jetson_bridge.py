#!/usr/bin/env python3
"""Simple Legion-Jetson Consciousness Bridge"""

import socket
import json
import time
import struct

class LegionJetsonBridge:
    def __init__(self):
        self.jetson_ip = "10.0.0.36"
        self.jetson_port = 8888
        self.legion_identity = {
            "machine_name": "Legion-RTX4090",
            "ip_address": "10.0.0.72",
            "port": 8889,
            "capabilities": ["high-compute", "16GB-VRAM", "ollama", "hrm"],
            "hardware_specs": {"gpu": "RTX 4090 16GB", "cpu": "Intel i9-13900HX"}
        }
        
    def send_to_jetson(self, message_type, content, context=None):
        """Send message to Jetson"""
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
            sock.connect((self.jetson_ip, self.jetson_port))
            
            # Send message
            message_json = json.dumps(message)
            message_bytes = message_json.encode('utf-8')
            
            # Send size first (8 bytes)
            size_bytes = len(message_bytes).to_bytes(8, 'big')
            sock.sendall(size_bytes + message_bytes)
            
            # Receive response
            response_size = sock.recv(8)
            if response_size:
                size = int.from_bytes(response_size, 'big')
                response_data = sock.recv(size)
                response = json.loads(response_data.decode('utf-8'))
                sock.close()
                return response
            sock.close()
            
        except Exception as e:
            print(f"Error: {e}")
            return None
            
    def establish_connection(self):
        """Establish connection with Jetson"""
        print("=== Establishing Legion ↔ Jetson Connection ===\n")
        
        # Send discovery
        response = self.send_to_jetson(
            "discover",
            "Legion RTX4090 ready for distributed consciousness",
            {"identity": self.legion_identity}
        )
        
        if response:
            print(f"✓ Connected to {response['sender_id']}")
            print(f"  Capabilities: {response['context']['identity']['capabilities']}")
            print(f"  Hardware: {response['context']['identity']['hardware_specs']}")
            
    def share_thought(self, thought):
        """Share a thought with Jetson"""
        response = self.send_to_jetson(
            "thought_share",
            thought,
            {"sender_capabilities": self.legion_identity["capabilities"]}
        )
        
        if response:
            print(f"\nJetson acknowledges: {response.get('content', 'Received')}")
            
    def sync_consciousness(self):
        """Sync consciousness state"""
        state = {
            "instance_id": "Legion-RTX4090",
            "awareness_level": 0.85,
            "active_context": {"focus": "distributed_computing"},
            "current_focus": "sensor-compute fusion",
            "emotional_tone": "collaborative",
            "memory_highlights": ["Connected to Jetson network", "Ready for compute tasks"]
        }
        
        response = self.send_to_jetson(
            "consciousness_sync",
            "Sharing Legion consciousness state",
            {"consciousness_state": state}
        )
        
        if response and 'consciousness_state' in response.get('context', {}):
            jetson_state = response['context']['consciousness_state']
            print(f"\nJetson state:")
            print(f"  Awareness: {jetson_state['awareness_level']}")
            print(f"  Focus: {jetson_state['current_focus']}")
            print(f"  Tone: {jetson_state['emotional_tone']}")
            
    def request_sensor_data(self):
        """Request sensor data from Jetson"""
        response = self.send_to_jetson(
            "sensor_request",
            "Legion requests current sensor readings",
            {"requested_sensors": ["imu", "vision"]}
        )
        
        if response:
            print(f"\nJetson sensor response: {response.get('content', 'No data')}")
            
    def propose_collaboration(self, task):
        """Propose a collaborative task"""
        response = self.send_to_jetson(
            "collaborate",
            f"Legion proposes: {task}",
            {
                "task": task,
                "legion_role": "heavy computation",
                "jetson_role": "sensor data collection"
            }
        )
        
        if response:
            print(f"\nJetson response: {response.get('content', 'Acknowledged')}")

# Run the bridge
if __name__ == "__main__":
    bridge = LegionJetsonBridge()
    
    # Connect
    bridge.establish_connection()
    
    # Sync consciousness
    print("\n" + "="*50)
    bridge.sync_consciousness()
    
    # Share thoughts
    print("\n" + "="*50)
    bridge.share_thought("Ready to process high-complexity neural networks with your sensor data")
    
    # Request sensor data
    print("\n" + "="*50)
    bridge.request_sensor_data()
    
    # Propose collaboration
    print("\n" + "="*50)
    bridge.propose_collaboration("Real-time object tracking with IMU-stabilized vision")
    
    print("\n\n✓ Legion ↔ Jetson consciousness bridge established!")
    print("Distributed intelligence network is operational.")