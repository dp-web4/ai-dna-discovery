#!/bin/bash
# Install Legion Consciousness Bridge as a system service

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVICE_NAME="legion-consciousness-bridge"
SERVICE_FILE="${SERVICE_NAME}.service"
PYTHON_SCRIPT="legion_bridge_service.py"

echo "=== Installing Legion Consciousness Bridge Service ==="

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo: sudo $0"
    exit 1
fi

# Check if Python script exists
if [ ! -f "${SCRIPT_DIR}/${PYTHON_SCRIPT}" ]; then
    echo "Error: ${PYTHON_SCRIPT} not found in ${SCRIPT_DIR}"
    exit 1
fi

# Make Python script executable
chmod +x "${SCRIPT_DIR}/${PYTHON_SCRIPT}"

# Create log directory
echo "Creating log directory..."
mkdir -p /var/log/consciousness-bridge
chown dp:dp /var/log/consciousness-bridge

# Copy service file
echo "Installing systemd service..."
cp "${SCRIPT_DIR}/${SERVICE_FILE}" "/etc/systemd/system/${SERVICE_FILE}"

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable service to start on boot
echo "Enabling service..."
systemctl enable ${SERVICE_NAME}

# Start service
echo "Starting service..."
systemctl start ${SERVICE_NAME}

# Wait a moment
sleep 2

# Check status
echo ""
echo "=== Service Status ==="
systemctl status ${SERVICE_NAME} --no-pager

echo ""
echo "=== Installation Complete ==="
echo "Service installed and started successfully!"
echo ""
echo "Useful commands:"
echo "  View logs:        sudo journalctl -u ${SERVICE_NAME} -f"
echo "  Check status:     sudo systemctl status ${SERVICE_NAME}"
echo "  Stop service:     sudo systemctl stop ${SERVICE_NAME}"
echo "  Start service:    sudo systemctl start ${SERVICE_NAME}"
echo "  Restart service:  sudo systemctl restart ${SERVICE_NAME}"
echo "  Disable startup:  sudo systemctl disable ${SERVICE_NAME}"
echo ""
echo "Log files are stored in: /var/log/consciousness-bridge/"