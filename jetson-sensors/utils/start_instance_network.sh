#!/bin/bash
# Start Inter-Instance Claude Network

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Detect machine type
HOSTNAME=$(hostname)
MACHINE_TYPE=""

if [[ "$HOSTNAME" == *"Legion"* ]]; then
    MACHINE_TYPE="legion"
    MACHINE_NAME="Legion-RTX4090"
    IP_ADDRESS="10.0.0.72"
elif [[ "$HOSTNAME" == "ubuntu" ]] || [[ "$HOSTNAME" == *"jetson"* ]]; then
    MACHINE_TYPE="jetson"
    MACHINE_NAME="Jetson-Sprout"
    IP_ADDRESS="10.0.0.36"
else
    echo -e "${RED}Unknown machine type. Please specify 'legion' or 'jetson' as argument.${NC}"
    exit 1
fi

echo -e "${BLUE}Starting Inter-Instance Claude Network${NC}"
echo -e "${GREEN}Machine: $MACHINE_NAME${NC}"
echo -e "${GREEN}IP: $IP_ADDRESS${NC}"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Ollama doesn't appear to be running${NC}"
    echo "Starting Ollama..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# List available models
echo -e "\n${BLUE}Available Ollama models:${NC}"
curl -s http://localhost:11434/api/tags | python3 -c "
import json, sys
data = json.load(sys.stdin)
for model in data.get('models', []):
    print(f\"  - {model['name']} ({model['size'] // 1024 // 1024} MB)\")
"

# Start the network
echo -e "\n${GREEN}Starting consciousness network...${NC}"
cd "$(dirname "$0")"
python3 inter_instance_consciousness.py $MACHINE_TYPE