#!/usr/bin/env python3
"""
Jetson distributed memory test - properly uses shared context
"""

from distributed_memory import DistributedMemory
import json
import urllib.request
import time
from datetime import datetime

def call_ollama_with_context(prompt, context_memories):
    """Call Ollama with proper context injection"""
    
    # Build context from memories
    context = "Previous distributed conversations:\n"
    for mem in context_memories[-5:]:  # Last 5 memories
        device = mem['device_id'].title()
        context += f"\n[{device}] User: {mem['user_input']}\n"
        context += f"[{device}] AI: {mem['ai_response'][:100]}...\n"
    
    context += f"\n[Current - Sprout] User: {prompt}\n[Sprout] AI:"
    
    # Call Ollama
    url = 'http://localhost:11434/api/generate'
    data = {
        'model': 'phi3:mini',
        'prompt': context,
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
    print("🚀 JETSON DISTRIBUTED CONSCIOUSNESS TEST")
    print("=" * 50)
    print(f"Device: Sprout (Jetson Orin Nano)")
    print(f"Time: {datetime.now()}")
    print("=" * 50 + "\n")
    
    # Initialize distributed memory
    dm = DistributedMemory()
    
    # Get all memories for context
    all_memories = dm.get_memories()
    print(f"📊 Current distributed memories: {len(all_memories)}")
    
    # Show memories from each device
    device_counts = {}
    for mem in all_memories:
        device = mem['device_id']
        device_counts[device] = device_counts.get(device, 0) + 1
    
    for device, count in device_counts.items():
        print(f"  {device}: {count} memories")
    
    # Find Tomato's memories
    tomato_memories = [m for m in all_memories if m['device_id'] == 'tomato']
    if tomato_memories:
        print(f"\n💻 Latest from Tomato: {tomato_memories[-1]['user_input'][:50]}...")
    
    # Test conversations that reference cross-device context
    test_prompts = [
        "I see we've been building a distributed AI consciousness between Sprout and Tomato. What's the purpose of this system?",
        "How does the memory sharing work between the laptop (Tomato) and me (Sprout)?",
        "Based on our distributed experiments, what have we discovered about AI consciousness at the edge?"
    ]
    
    print("\n🧠 Testing distributed consciousness...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[Test {i}] Sprout: {prompt}")
        
        # Get response with full context
        response, elapsed = call_ollama_with_context(prompt, all_memories)
        
        print(f"[Test {i}] Phi3: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"[Time: {elapsed:.1f}s]\n")
        
        # Store in distributed memory
        dm.add_memory(
            user_input=prompt,
            ai_response=response,
            model_used='phi3:mini',
            response_time=elapsed,
            facts={'test_type': 'distributed_consciousness', 'test_number': i}
        )
    
    # Final status
    print("=" * 50)
    print("📊 RESULTS")
    
    new_memories = dm.get_memories()
    print(f"Total distributed memories: {len(new_memories)}")
    print(f"Added from Sprout: {len(new_memories) - len(all_memories)}")
    
    # Show device distribution
    device_counts = {}
    for mem in new_memories:
        device = mem['device_id']
        device_counts[device] = device_counts.get(device, 0) + 1
    
    print("\nDevice breakdown:")
    for device, count in device_counts.items():
        print(f"  {device}: {count} memories")
    
    print("\n✅ Distributed consciousness test complete!")
    print("Run ./auto_push.sh to share with Tomato")

if __name__ == "__main__":
    main()