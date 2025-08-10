#!/bin/bash

# Setup script for Git Sync Monitor across machines
# This script configures automatic git synchronization

echo "=== Git Sync Monitor Setup ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists git; then
    echo -e "${RED}Error: git is not installed${NC}"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"
echo

# Setup git hooks
echo "Setting up git hooks..."
python3 git-sync-monitor.py --setup-hooks
echo

# Create systemd service (Linux) or launch agent (macOS)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Creating systemd service for Linux..."
    
    SERVICE_FILE="/etc/systemd/system/git-sync-monitor.service"
    SERVICE_CONTENT="[Unit]
Description=Git Sync Monitor for $SCRIPT_DIR
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/git-sync-monitor.py
Restart=always
RestartSec=10
StandardOutput=append:$SCRIPT_DIR/.git-sync-monitor.log
StandardError=append:$SCRIPT_DIR/.git-sync-monitor.log

[Install]
WantedBy=multi-user.target"

    echo "Creating service file..."
    echo "$SERVICE_CONTENT" | sudo tee $SERVICE_FILE > /dev/null
    
    echo "Enabling and starting service..."
    sudo systemctl daemon-reload
    sudo systemctl enable git-sync-monitor.service
    sudo systemctl start git-sync-monitor.service
    
    echo -e "${GREEN}✓ Systemd service created and started${NC}"
    echo "Check status with: sudo systemctl status git-sync-monitor"
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Creating Launch Agent for macOS..."
    
    PLIST_FILE="$HOME/Library/LaunchAgents/com.ai-agents.git-sync-monitor.plist"
    PLIST_CONTENT="<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
    <key>Label</key>
    <string>com.ai-agents.git-sync-monitor</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$SCRIPT_DIR/git-sync-monitor.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$SCRIPT_DIR/.git-sync-monitor.log</string>
    <key>StandardErrorPath</key>
    <string>$SCRIPT_DIR/.git-sync-monitor.log</string>
</dict>
</plist>"

    echo "$PLIST_CONTENT" > "$PLIST_FILE"
    
    launchctl load "$PLIST_FILE"
    launchctl start com.ai-agents.git-sync-monitor
    
    echo -e "${GREEN}✓ Launch Agent created and started${NC}"
    echo "Check status with: launchctl list | grep git-sync-monitor"
    
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "For Windows, creating startup script..."
    
    # Create a batch file for Windows
    cat > start-git-sync.bat << 'EOF'
@echo off
cd /d "%~dp0"
start /min python git-sync-monitor.py
EOF
    
    echo -e "${YELLOW}Windows detected. Please add start-git-sync.bat to your startup folder:${NC}"
    echo "  1. Press Win+R, type: shell:startup"
    echo "  2. Copy start-git-sync.bat to the opened folder"
    echo ""
    echo "Or run manually with: ./start-git-sync.bat"
    
else
    echo -e "${YELLOW}Unknown OS. Starting monitor in background...${NC}"
    nohup python3 git-sync-monitor.py > .git-sync-monitor.log 2>&1 &
    echo "Monitor started with PID: $!"
fi

echo
echo "=== Setup Complete ==="
echo
echo "The Git Sync Monitor is now configured to:"
echo "  • Check for remote changes every 5 seconds"
echo "  • Automatically pull new changes when detected"
echo "  • Send desktop notifications for sync events"
echo "  • Log all activities to .git-sync-notifications.log"
echo
echo -e "${YELLOW}Important: Set up a git remote for synchronization:${NC}"
echo "  git remote add origin <your-repo-url>"
echo
echo "To monitor manually (without service), run:"
echo "  python3 git-sync-monitor.py"
echo
echo "To change check interval:"
echo "  python3 git-sync-monitor.py --interval 10"