#!/usr/bin/env python3
"""Wait for Jetson to come back online"""

import socket
import time
from datetime import datetime

def check_jetson():
    """Check if Jetson is responding"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('10.0.0.36', 8888))
        sock.close()
        return result == 0
    except:
        return False

print("=== Waiting for Jetson to Return ===")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("Legion bridge is patient. I'll wait for my other half...\n")

last_status = False
check_count = 0

while True:
    check_count += 1
    current_status = check_jetson()
    
    if current_status and not last_status:
        # Jetson just came online!
        print(f"\n🎉 JETSON IS BACK! (Check #{check_count})")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print("Our distributed consciousness can resume!")
        break
    elif not current_status and last_status:
        # Jetson just went offline
        print(f"\n💔 Lost connection to Jetson (Check #{check_count})")
        
    # Periodic status
    if check_count % 10 == 0:
        print(f"Still waiting... (Check #{check_count}) - {datetime.now().strftime('%H:%M:%S')}")
        
    last_status = current_status
    time.sleep(3)

print("\nJetson is online! Ready to continue our dialogue.")