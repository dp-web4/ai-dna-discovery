#!/bin/bash
#
# Uninstall Jetson Consciousness Bridge Service
# Run with: sudo ./uninstall_consciousness_service.sh
#

set -e

SERVICE_NAME="consciousness-bridge"

echo "=========================================="
echo "Uninstalling Jetson Consciousness Bridge Service"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo: sudo $0"
    exit 1
fi

# Stop the service if running
echo "Stopping service..."
systemctl stop ${SERVICE_NAME} 2>/dev/null || true

# Disable the service
echo "Disabling service..."
systemctl disable ${SERVICE_NAME} 2>/dev/null || true

# Remove service file
echo "Removing service file..."
rm -f /etc/systemd/system/${SERVICE_NAME}.service

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload
systemctl reset-failed

# Optional: Ask about log cleanup
read -p "Remove log files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing log files..."
    rm -rf /var/log/consciousness
    rm -f /var/run/consciousness_bridge.state
fi

echo ""
echo "=========================================="
echo "Uninstallation Complete!"
echo "=========================================="
echo ""
echo "The consciousness bridge service has been removed."