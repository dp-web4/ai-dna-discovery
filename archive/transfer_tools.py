#!/usr/bin/env python3
"""
Easy transfer tools for Sprout-Tomato communication
"""

import subprocess
import os
import sys

SPROUT_IP = "10.0.0.36"
TOMATO_IP = None  # Will be set when known

def send_to_tomato(filename, tomato_ip):
    """Send file to Tomato using scp"""
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return
    
    print(f"📤 Sending {filename} to Tomato ({tomato_ip})...")
    cmd = f"scp {filename} dp@{tomato_ip}:~/ai-workspace/ai-dna-discovery/"
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("✅ Transfer complete!")
    else:
        print("❌ Transfer failed")

def get_from_tomato(filename, tomato_ip):
    """Get file from Tomato using scp"""
    print(f"📥 Getting {filename} from Tomato ({tomato_ip})...")
    cmd = f"scp dp@{tomato_ip}:~/ai-workspace/ai-dna-discovery/{filename} ."
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("✅ Transfer complete!")
    else:
        print("❌ Transfer failed")

def sync_model_training(tomato_ip):
    """Sync the entire model-training directory"""
    print("🔄 Syncing model-training directory...")
    cmd = f"rsync -avz --progress dp@{tomato_ip}:~/ai-workspace/ai-dna-discovery/model-training/ ./model-training/"
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("✅ Sync complete!")
    else:
        print("❌ Sync failed")

def quick_share(filename):
    """Quick share using Python HTTP server"""
    print(f"🌐 Sharing {filename} via HTTP...")
    print(f"📋 From Tomato, run:")
    print(f"   wget http://{SPROUT_IP}:8080/{filename}")
    print(f"   or")
    print(f"   curl -O http://{SPROUT_IP}:8080/{filename}")
    print("\nPress Ctrl+C to stop sharing")
    
    # Start simple HTTP server
    subprocess.run(["python3", "-m", "http.server", "8080"])

if __name__ == "__main__":
    print("🚀 Sprout-Tomato Transfer Tools")
    print(f"📍 Sprout IP: {SPROUT_IP}")
    print("\nUsage:")
    print("  python3 transfer_tools.py send <file> <tomato_ip>")
    print("  python3 transfer_tools.py get <file> <tomato_ip>")
    print("  python3 transfer_tools.py sync <tomato_ip>")
    print("  python3 transfer_tools.py share <file>")
    
    if len(sys.argv) < 2:
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "share" and len(sys.argv) >= 3:
        quick_share(sys.argv[2])
    elif cmd == "send" and len(sys.argv) >= 4:
        send_to_tomato(sys.argv[2], sys.argv[3])
    elif cmd == "get" and len(sys.argv) >= 4:
        get_from_tomato(sys.argv[2], sys.argv[3])
    elif cmd == "sync" and len(sys.argv) >= 3:
        sync_model_training(sys.argv[2])
    else:
        print("❌ Invalid command")