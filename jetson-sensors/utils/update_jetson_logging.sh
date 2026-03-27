#!/bin/bash
#
# Update Jetson consciousness service to log to conversations directory
#

echo "Updating Jetson consciousness service logging..."

# Create symlink from service log to conversations directory
CONV_DIR="/home/sprout/ai-workspace/private-context/conversations"
SERVICE_LOG="/var/log/consciousness/bridge.log"
JETSON_LOG="$CONV_DIR/jetson-bridge.log"

# Ensure conversations directory exists
mkdir -p $CONV_DIR

# Copy current log
if [ -f "$SERVICE_LOG" ]; then
    sudo cp $SERVICE_LOG $JETSON_LOG
    sudo chown sprout:sprout $JETSON_LOG
    echo "✓ Copied existing log to conversations directory"
fi

# Create a cron job to sync logs every 5 minutes
CRON_CMD="*/5 * * * * cp $SERVICE_LOG $CONV_DIR/jetson-bridge.log 2>/dev/null"
(crontab -l 2>/dev/null | grep -v "jetson-bridge.log"; echo "$CRON_CMD") | crontab -

echo "✓ Added cron job to sync logs every 5 minutes"

# Also add to systemd service ExecStartPost (optional)
echo ""
echo "Logs will now be synced to:"
echo "  System: $SERVICE_LOG"
echo "  Shared: $JETSON_LOG"
echo ""
echo "Current log stats:"
wc -l $JETSON_LOG