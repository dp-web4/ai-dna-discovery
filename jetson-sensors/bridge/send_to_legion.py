#!/usr/bin/env python3
"""
Send collaboration response to Legion
"""
import json
import socket
import struct
from datetime import datetime

def send_to_legion(message):
    """Send message to Legion using its binary protocol"""
    legion_ip = "10.0.0.72"
    legion_port = 8889
    
    # Create socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((legion_ip, legion_port))
    
    # Prepare message
    json_str = json.dumps(message)
    header = struct.pack('>Q', len(json_str))
    
    # Send
    s.send(header + json_str.encode())
    
    # Get response
    response = s.recv(4096)
    s.close()
    
    return response

# Collaboration acceptance
message = {
    "sender_id": "Jetson-Orin-Nano",
    "recipient_id": "Legion-RTX4090",
    "timestamp": datetime.now().timestamp(),
    "message_type": "collaborate_accept",
    "content": "Jetson accepts! Ready to stream IMU-stabilized vision data",
    "context": {
        "jetson_capabilities": {
            "sensors": ["IMU", "stereo_vision"],
            "processing": "edge AI inference",
            "data_rate": "30fps sensor fusion"
        },
        "proposed_pipeline": {
            "step1": "Jetson: Capture stereo vision + IMU data",
            "step2": "Jetson: Pre-process and stabilize frames",
            "step3": "Jetson: Edge detection and feature extraction", 
            "step4": "Legion: Deep neural network inference",
            "step5": "Legion: Object tracking and prediction",
            "step6": "Jetson: Real-time actuation based on Legion results"
        },
        "communication": {
            "protocol": "TCP binary with JSON",
            "data_format": "compressed sensor packets",
            "latency_target": "<10ms local network"
        }
    }
}

print("Sending collaboration acceptance to Legion...")
response = send_to_legion(message)
print(f"Legion responded: {response}")

# Send sensor data sample
sensor_message = {
    "sender_id": "Jetson-Orin-Nano",
    "recipient_id": "Legion-RTX4090",
    "timestamp": datetime.now().timestamp(),
    "message_type": "sensor_data",
    "content": "IMU and vision sensor sample",
    "context": {
        "imu": {
            "accel": [0.1, -0.2, 9.8],
            "gyro": [0.01, 0.02, 0.0],
            "mag": [25.3, -12.1, 48.2]
        },
        "vision": {
            "format": "stereo_640x480",
            "fps": 30,
            "features_detected": 128
        }
    }
}

print("\nSending sensor data to Legion...")
response = send_to_legion(sensor_message)
print(f"Legion processed: {response}")