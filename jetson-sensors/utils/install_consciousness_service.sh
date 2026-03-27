#!/bin/bash
#
# Install Jetson Consciousness Bridge as systemd service
# Run with: sudo ./install_consciousness_service.sh
#

set -e

SERVICE_NAME="consciousness-bridge"
SERVICE_FILE="${SERVICE_NAME}.service"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="/var/log/consciousness"

echo "=========================================="
echo "Installing Jetson Consciousness Bridge Service"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo: sudo $0"
    exit 1
fi

# Create log directory
echo "Creating log directory: ${LOG_DIR}"
mkdir -p ${LOG_DIR}
chown sprout:sprout ${LOG_DIR}

# Create state directory
echo "Creating state directory: /var/run"
touch /var/run/consciousness_bridge.state
chown sprout:sprout /var/run/consciousness_bridge.state

# Make service script executable
echo "Setting permissions on service script..."
chmod +x ${SCRIPT_DIR}/jetson_consciousness_service.py

# Copy service file to systemd
echo "Installing systemd service..."
cp ${SCRIPT_DIR}/${SERVICE_FILE} /etc/systemd/system/

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable service to start on boot
echo "Enabling service to start on boot..."
systemctl enable ${SERVICE_NAME}

# Start the service
echo "Starting service..."
systemctl start ${SERVICE_NAME}

# Wait a moment for service to start
sleep 2

# Check status
echo ""
echo "=========================================="
echo "Service Status:"
echo "=========================================="
systemctl status ${SERVICE_NAME} --no-pager

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  View status:  sudo systemctl status ${SERVICE_NAME}"
echo "  View logs:    sudo journalctl -u ${SERVICE_NAME} -f"
echo "  View log file: tail -f ${LOG_DIR}/bridge.log"
echo "  Restart:      sudo systemctl restart ${SERVICE_NAME}"
echo "  Stop:         sudo systemctl stop ${SERVICE_NAME}"
echo "  Disable:      sudo systemctl disable ${SERVICE_NAME}"
echo ""
echo "The consciousness bridge is now running and will start automatically on boot!"