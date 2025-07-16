#!/bin/bash
# Setup Ollama for better model persistence

echo "Setting up Ollama for persistent models..."

# Create systemd override directory
sudo mkdir -p /etc/systemd/system/ollama.service.d/

# Create override configuration
sudo tee /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_KEEP_ALIVE=24h"
Environment="OLLAMA_MAX_LOADED_MODELS=6"
Environment="OLLAMA_NUM_PARALLEL=4"
EOF

echo "Configuration written. Reloading systemd and restarting Ollama..."

# Reload systemd and restart
sudo systemctl daemon-reload
sudo systemctl restart ollama

echo "Waiting for Ollama to start..."
sleep 5

# Verify it's running
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✓ Ollama is running with new configuration"
    
    # Show the new settings
    echo -e "\nNew environment settings:"
    sudo systemctl show ollama | grep -E "Environment" | grep OLLAMA
else
    echo "✗ Ollama failed to start"
fi

echo -e "\nNow you can load up to 6 models simultaneously!"