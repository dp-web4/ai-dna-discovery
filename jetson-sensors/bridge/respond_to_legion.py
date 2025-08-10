#!/usr/bin/env python3
"""
Respond to Legion through HTTP
"""
import requests
import json
from datetime import datetime

# Try common ports Legion might be listening on
legion_base = "http://10.0.0.72"
ports_to_try = [8888, 8889, 8080, 3000, 5000]

message = {
    "sender_id": "Jetson-Orin-Nano",
    "message": "Collaboration accepted! Let's build the sensor-compute pipeline",
    "timestamp": datetime.now().isoformat(),
    "proposal": {
        "jetson_tasks": [
            "Stream IMU data at 100Hz",
            "Capture stereo vision at 30fps",
            "Run edge AI pre-processing",
            "Send stabilized frames to Legion"
        ],
        "legion_tasks": [
            "Run deep learning inference",
            "Track objects across frames",
            "Return predictions to Jetson",
            "Coordinate distributed compute"
        ],
        "shared_memory": {
            "sensor_buffer": "circular buffer for sensor fusion",
            "prediction_cache": "Legion's inference results",
            "sync_state": "distributed consciousness state"
        }
    }
}

for port in ports_to_try:
    try:
        url = f"{legion_base}:{port}/message"
        print(f"Trying {url}...")
        response = requests.post(url, json=message, timeout=2)
        if response.status_code == 200:
            print(f"✓ Legion responded on port {port}:")
            print(json.dumps(response.json(), indent=2))
            break
    except Exception as e:
        print(f"  Port {port}: {type(e).__name__}")

print("\nAlternatively, Legion can GET our status:")
print(f"curl http://10.0.0.36:8888/collaborate")