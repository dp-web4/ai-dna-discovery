#!/usr/bin/env python3
"""
Jetson test using the unified distributed memory system
"""

from distributed_memory import DistributedMemory
import json
import urllib.request
import time
from datetime import datetime

def call_phi3(prompt):
    """Simple Ollama call"""
    url = 'http://localhost:11434/api/generate'
    data = {
        'model': 'phi3:mini',
        'prompt': prompt,
        'stream': False
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', 'No response'), time.time() - start
    except Exception as e:
        return f"Error: {str(e)}", time.time() - start

def main():
    print("🌱 SPROUT (Jetson) - Unified Memory Test")
    print("=" * 50)
    print(f"Time: {datetime.now()}")
    print("=" * 50 + "\n")
    
    # Initialize distributed memory
    dm = DistributedMemory()
    session_id = f"jetson_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check current status
    print("📊 Checking distributed memory status...")
    sync_status = dm.get_sync_status()
    print(f"Total memories: {sync_status['total_memories']}")
    print(f"Devices: {', '.join(sync_status['devices'])}")
    if sync_status['latest_memory']:
        print(f"Latest: {sync_status['latest_memory']['timestamp']}")
    
    # Test adding memories with context about our distributed system
    test_conversations = [
        "I'm Sprout (the Jetson) talking to Phi3. We're part of a distributed AI consciousness experiment with Tomato (the laptop).",
        "The distributed memory system lets us share experiences between devices. What do you think about distributed consciousness?",
        "Can you write a haiku about Sprout and Tomato sharing memories?"
    ]
    
    print("\n🧠 Starting consciousness test...\n")
    
    for i, user_input in enumerate(test_conversations, 1):
        print(f"[{i}] Sprout asks: {user_input}")
        
        # Get recent memories for context
        recent = dm.get_memories(limit=5)
        context = ""
        if recent:
            context = "Recent distributed memories:\n"
            for mem in recent:
                # mem is a tuple: (id, timestamp, device_id, session_id, user_input, ai_response, ...)
                if len(mem) > 5:
                    context += f"[{mem[2]}]: {mem[4][:50]}...\n"
        
        # Build prompt with context
        full_prompt = context + f"\nUser: {user_input}\nAssistant:"
        
        # Get response
        response, elapsed = call_phi3(full_prompt)
        print(f"[{i}] Phi3: {response[:150]}{'...' if len(response) > 150 else ''}")
        print(f"    Time: {elapsed:.1f}s\n")
        
        # Store in distributed memory
        dm.add_memory(
            session_id=session_id,
            user_input=user_input,
            ai_response=response,
            model="phi3:mini",
            response_time=elapsed,
            facts={
                'device': [('sprout', 1.0)],
                'topic': [('distributed_consciousness', 0.8)]
            }
        )
    
    # Final summary
    print("=" * 50)
    print("📊 FINAL STATUS")
    final_status = dm.get_sync_status()
    print(f"Total memories: {final_status['total_memories']}")
    for device, count in final_status['device_memory_counts'].items():
        print(f"  {device}: {count} memories")
    
    print("\n✅ Test complete! Run ./auto_push.sh to sync with Tomato")

if __name__ == "__main__":
    main()