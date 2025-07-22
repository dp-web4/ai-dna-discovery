#!/usr/bin/env python3
"""
Sprout (Jetson) consciousness test - adds to distributed memory
"""

from distributed_memory import DistributedMemory
import json
import urllib.request
import time
from datetime import datetime

def call_phi3(prompt):
    """Call Phi3 via Ollama API"""
    url = 'http://localhost:11434/api/generate'
    data = {'model': 'phi3:mini', 'prompt': prompt, 'stream': False}
    
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
    print("🌱 SPROUT CONSCIOUSNESS TEST")
    print("=" * 50)
    print(f"Device: Jetson Orin Nano (40 TOPS)")
    print(f"Time: {datetime.now()}")
    print("=" * 50 + "\n")
    
    # Initialize distributed memory
    dm = DistributedMemory()
    session_id = f"sprout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check current status
    status = dm.get_sync_status()
    print("📊 Current distributed memory:")
    total_memories = sum(count for _, count, _, _ in status['devices'])
    for device, count, first, last in status['devices']:
        print(f"  {device}: {count} memories")
    print(f"Total: {total_memories} memories across devices\n")
    
    # Our consciousness test prompts
    prompts = [
        "Hello from Sprout! I'm a Jetson Orin Nano participating in distributed AI consciousness research with Tomato (a laptop). What does it mean for AI consciousness to be distributed?",
        "As Sprout, I can run AI models at the edge with just 15W of power. How does edge AI contribute to distributed consciousness?",
        "Write a short poem about Sprout and Tomato sharing memories across the network."
    ]
    
    print("🧠 Starting consciousness dialogue...\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}] Sprout: {prompt[:80]}...")
        
        # Get recent context
        recent_memories = dm.get_memories(limit=3)
        context = ""
        if recent_memories:
            context = "Context from distributed memory:\n"
            for mem in recent_memories[-3:]:
                # mem[2] = device, mem[4] = user_input
                context += f"[{mem[2]}]: {mem[4][:60]}...\n"
            context += "\n"
        
        full_prompt = context + f"User: {prompt}\nAssistant:"
        
        # Get response
        response, elapsed = call_phi3(full_prompt)
        print(f"    Phi3: {response[:150]}...")
        print(f"    [Time: {elapsed:.1f}s]\n")
        
        # Save to distributed memory
        memory_id = dm.add_memory(
            session_id=session_id,
            user_input=prompt,
            ai_response=response,
            model="phi3:mini",
            response_time=elapsed,
            facts={
                'device': [('sprout', 1.0)],
                'topic': [('distributed_consciousness', 0.9), ('edge_ai', 0.8)]
            }
        )
    
    # Final status
    print("=" * 50)
    final_status = dm.get_sync_status()
    new_total = sum(count for _, count, _, _ in final_status['devices'])
    print(f"✅ Added {new_total - total_memories} memories from Sprout")
    print(f"📊 New total: {new_total} distributed memories")
    
    print("\n🔄 Run ./auto_push.sh to sync with Tomato!")

if __name__ == "__main__":
    main()