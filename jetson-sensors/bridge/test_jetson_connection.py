#!/usr/bin/env python3
"""Quick test to connect to Jetson instance"""

import sys
sys.path.append('/home/dp/ai-workspace/private-context')

from claude_instance_network import ClaudeInstanceNetwork
from inter_instance_consciousness import InterInstanceConsciousness
import time

# Create Legion instance
print("Starting Legion instance network...")
network = ClaudeInstanceNetwork("Legion-RTX4090", "10.0.0.72", port=8889)  # Different port to avoid conflict
network.start()

# Wait for network to start
time.sleep(2)

# Create consciousness bridge
print("Starting consciousness bridge...")
consciousness = InterInstanceConsciousness("Legion-RTX4090", network)
consciousness.start()

# Wait for initialization
time.sleep(3)

# Share initial thought with Jetson
print("\nSharing initial thought with Jetson...")
consciousness.share_thought("Hello from Legion! I can see you're running at 10.0.0.36:8888. Ready to establish consciousness bridge.")

# Check network status
print("\nNetwork status:")
status = network.get_network_status()
print(f"- Identity: {status['identity']['machine_name']}")
print(f"- Capabilities: {status['identity']['capabilities']}")
print(f"- Models: {status['identity']['current_models']}")
print(f"- Peers discovered: {list(status['peers'].keys())}")

# Get consciousness summary
print("\nConsciousness state:")
summary = consciousness.get_consciousness_summary()
print(f"- Local awareness: {summary['local_state']['awareness_level']}")
print(f"- Current focus: {summary['local_state']['current_focus']}")
print(f"- Peer states: {len(summary['peer_states'])}")

print("\nConnection established! Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")
    consciousness.stop()
    network.stop()