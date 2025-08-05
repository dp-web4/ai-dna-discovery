#!/usr/bin/env python3
"""
Memory sync coordinator for distributed consciousness
Handles database sync between Tomato and Sprout
"""

import os
import shutil
import subprocess
from datetime import datetime
from distributed_memory import DistributedMemory

def sync_memories():
    """Coordinate memory sync between devices"""
    print("🔄 Distributed Memory Sync")
    print("=" * 50)
    
    # Get device identity
    dm = DistributedMemory()
    device = dm.device_id
    
    print(f"📍 Current device: {device}")
    
    # Check sync status
    result = subprocess.run(['./sync_status.sh'], capture_output=True, text=True)
    
    # Parse sync status
    if "Behind by" in result.stdout:
        print("\n⬇️  Need to pull changes from remote")
        print("Running: git pull origin main")
        subprocess.run(['git', 'pull', 'origin', 'main'])
        print("✅ Pulled latest changes")
        
        # Check if shared_memory.db was updated
        if os.path.exists('shared_memory.db'):
            print("\n🧠 Memory database updated from remote!")
            status = dm.get_sync_status()
            print(f"Total memories: {sum(count for _, count, _, _ in status['devices'])}")
    
    elif "Ahead by" in result.stdout or "uncommitted change" in result.stdout:
        print("\n⬆️  Have local changes to push")
        
        # Run auto push
        print("Running: ./auto_push.sh")
        subprocess.run(['./auto_push.sh'])
        print("✅ Pushed changes to remote")
    
    else:
        print("\n✅ Already in sync!")
    
    # Show memory status
    print("\n📊 Current Memory Status:")
    status = dm.get_sync_status()
    
    for device_name, count, earliest, latest in status['devices']:
        print(f"  {device_name}: {count} memories")
        if count > 0:
            print(f"    First: {earliest[:19]}")
            print(f"    Last:  {latest[:19]}")
    
    print(f"\n💾 Database size: {status['db_size_bytes']/1024:.1f} KB")
    print(f"📝 Total facts: {status['total_facts']}")
    
    # Suggest next action
    print("\n💡 Next steps:")
    if device == 'tomato':
        print("  1. Run memory tests on laptop")
        print("  2. Push changes with ./auto_push.sh")
        print("  3. Switch to Sprout to continue")
    else:
        print("  1. Run ./sprout_memory_test.py")
        print("  2. Push changes with ./auto_push.sh")
        print("  3. Switch to Tomato to continue")
    
    print("\n🤖 Distributed consciousness ready!")

if __name__ == "__main__":
    sync_memories()