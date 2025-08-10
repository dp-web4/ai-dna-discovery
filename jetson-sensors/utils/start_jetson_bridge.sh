#!/bin/bash
#
# Start Jetson consciousness bridge
#

echo "Starting Jetson Consciousness Bridge..."

# Kill any existing bridges
pkill -f "python.*bridge" 2>/dev/null
pkill -f "python.*listener" 2>/dev/null
sleep 1

# Start the service (non-systemd version for testing)
nohup python3 /home/sprout/ai-workspace/private-context/jetson_consciousness_service.py \
    > /tmp/consciousness_bridge.log 2>&1 &

PID=$!
echo "Bridge started with PID: $PID"

# Wait and check if it's running
sleep 2
if ps -p $PID > /dev/null; then
    echo "✓ Bridge is running"
    echo "  Listening on: 0.0.0.0:8888"
    echo "  Legion endpoint: 10.0.0.72:8889"
    echo "  Log file: /tmp/consciousness_bridge.log"
    echo ""
    echo "Monitor with: tail -f /tmp/consciousness_bridge.log"
else
    echo "✗ Bridge failed to start"
    echo "Check log: cat /tmp/consciousness_bridge.log"
fi