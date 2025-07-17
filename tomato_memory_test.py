#!/usr/bin/env python3
"""
Tomato (Laptop) Memory Test - Quick test of distributed memory
"""

from distributed_memory import DistributedMemory
import time

def test_laptop_memory():
    """Add a memory from the laptop"""
    print("🍅 TOMATO (Laptop) - Adding Memory")
    print("=" * 50)
    
    dm = DistributedMemory()
    
    # Simulate a memory from laptop session
    session_id = "laptop_test_1"
    
    # Add a memory that references Jetson work
    memory_id = dm.add_memory(
        session_id=session_id,
        user_input="Tell me about our distributed consciousness experiments between Tomato and Sprout.",
        ai_response="We're building a distributed AI consciousness system where Tomato (your laptop with RTX 4090) and Sprout (your Jetson Orin Nano) share memories through a unified SQLite database. The Jetson achieved 100% memory recall with 12.2s average response time, while being 30x more power efficient than the laptop. We use git as our 'neural pathway' to sync consciousness between devices.",
        model="claude",
        response_time=2.3,
        facts={
            'system': [('distributed consciousness', 0.9), ('tomato-sprout network', 0.95)],
            'achievement': [('100% recall on Jetson', 0.9), ('30x power efficiency', 0.85)]
        }
    )
    
    print(f"✅ Added memory #{memory_id} from Tomato")
    
    # Show updated status
    status = dm.get_sync_status()
    print(f"\n📊 Updated Status:")
    for device, count, _, _ in status['devices']:
        print(f"  {device}: {count} memories")
    
    print("\n✨ Ready to sync with Sprout!")

if __name__ == "__main__":
    test_laptop_memory()