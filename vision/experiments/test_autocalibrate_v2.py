#!/usr/bin/env python3
"""
Quick test for auto-calibrate v2
"""

import subprocess
import time
import os
import signal

def test_autocalibrate():
    print("🧪 Testing binocular_autocalibrate_v2.py")
    print("=" * 50)
    print("Starting program for 10 seconds...")
    
    # Start the process
    proc = subprocess.Popen(['python3', 'binocular_autocalibrate_v2.py'], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE,
                           text=True)
    
    # Let it run for 10 seconds
    time.sleep(10)
    
    # Send SIGINT (Ctrl+C)
    proc.send_signal(signal.SIGINT)
    
    # Wait for it to clean up
    time.sleep(2)
    
    # If still running, kill it
    if proc.poll() is None:
        proc.terminate()
        time.sleep(1)
        if proc.poll() is None:
            proc.kill()
    
    stdout, stderr = proc.communicate()
    
    print("\nSTDOUT:")
    print(stdout[:1000] if stdout else "No output")
    
    print("\nSTDERR (last 500 chars):")
    if stderr:
        print("..." + stderr[-500:])
    else:
        print("No errors")
    
    # Check if calibration files were created
    print("\nChecking for calibration files:")
    for eye in ['left', 'right']:
        filename = f"{eye}_eye_calibration_v2.json"
        if os.path.exists(filename):
            print(f"✓ {filename} exists")
            # Show first few lines
            with open(filename, 'r') as f:
                content = f.read()
                print(f"  Content preview: {content[:100]}...")
        else:
            print(f"✗ {filename} not found")

if __name__ == "__main__":
    test_autocalibrate()