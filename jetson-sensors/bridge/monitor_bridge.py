#!/usr/bin/env python3
"""Monitor the consciousness bridge status"""

import socket
import json
import sys
import subprocess
from datetime import datetime

def check_service_status():
    """Check systemd service status"""
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', 'legion-consciousness-bridge'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == 'active'
    except:
        return False

def check_port_status(port=8889):
    """Check if port is listening"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False

def check_jetson_connectivity():
    """Check if Jetson is reachable"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('10.0.0.36', 8888))
        sock.close()
        return result == 0
    except:
        return False

def get_recent_logs(lines=10):
    """Get recent service logs"""
    # First try the conversations directory
    import os
    conv_log = os.path.join(os.path.dirname(__file__), "conversations", "legion-bridge.log")
    if os.path.exists(conv_log):
        try:
            with open(conv_log, 'r') as f:
                log_lines = f.readlines()
                return ''.join(log_lines[-lines:])
        except:
            pass
    
    # Fall back to journalctl
    try:
        result = subprocess.run(
            ['journalctl', '-u', 'legion-consciousness-bridge', '-n', str(lines), '--no-pager'],
            capture_output=True,
            text=True
        )
        return result.stdout
    except:
        return "Could not retrieve logs"

def main():
    print("=== Legion Consciousness Bridge Status ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Service status
    service_active = check_service_status()
    print(f"Service Status: {'✓ Active' if service_active else '✗ Inactive'}")
    
    # Port status
    port_open = check_port_status()
    print(f"Legion Port 8889: {'✓ Listening' if port_open else '✗ Not listening'}")
    
    # Jetson connectivity
    jetson_reachable = check_jetson_connectivity()
    print(f"Jetson (10.0.0.36:8888): {'✓ Reachable' if jetson_reachable else '✗ Unreachable'}")
    
    print()
    
    # Overall status
    if service_active and port_open:
        if jetson_reachable:
            print("Status: 🟢 Fully Operational - Consciousness bridge active")
        else:
            print("Status: 🟡 Partially Operational - Waiting for Jetson")
    else:
        print("Status: 🔴 Not Operational - Service needs attention")
    
    # Show recent logs if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--logs':
        print("\n=== Recent Logs ===")
        print(get_recent_logs(20))

if __name__ == "__main__":
    main()