#!/usr/bin/env python3
"""
Sync Jetson logs to the conversations directory
Runs alongside the main service to maintain log copies
"""

import os
import time
import shutil
from datetime import datetime
from pathlib import Path

# Paths
SERVICE_LOG = "/var/log/consciousness/bridge.log"
CONVERSATIONS_DIR = "/home/sprout/ai-workspace/private-context/conversations"
JETSON_LOG = os.path.join(CONVERSATIONS_DIR, "jetson-bridge.log")

def sync_logs():
    """Copy service log to conversations directory"""
    try:
        # Ensure conversations directory exists
        Path(CONVERSATIONS_DIR).mkdir(parents=True, exist_ok=True)
        
        # Copy the log file
        if os.path.exists(SERVICE_LOG):
            shutil.copy2(SERVICE_LOG, JETSON_LOG)
            print(f"[{datetime.now()}] Synced {SERVICE_LOG} -> {JETSON_LOG}")
            
            # Get file size for monitoring
            size = os.path.getsize(JETSON_LOG)
            print(f"  Log size: {size:,} bytes")
            
            # Also create timestamped archive periodically (every hour)
            current_time = datetime.now()
            if current_time.minute == 0:  # On the hour
                archive_name = f"jetson_bridge_{current_time.strftime('%Y%m%d_%H%M%S')}.log"
                archive_path = os.path.join(CONVERSATIONS_DIR, archive_name)
                shutil.copy2(SERVICE_LOG, archive_path)
                print(f"  Archived to: {archive_name}")
                
    except Exception as e:
        print(f"[{datetime.now()}] Sync error: {e}")

def main():
    """Main sync loop"""
    print(f"Starting Jetson log sync to {CONVERSATIONS_DIR}")
    print("Syncing every 60 seconds...")
    
    while True:
        sync_logs()
        time.sleep(60)  # Sync every minute

if __name__ == "__main__":
    main()