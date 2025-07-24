#!/usr/bin/env python3
"""
Sprout (Jetson) Memory Test - For distributed consciousness
Tests loading memories from Tomato and adding new ones
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distributed_memory import DistributedMemory
import json
import time
import urllib.request

def chat_with_ollama(model, prompt):
    """Simple Ollama API call"""
    url = 'http://localhost:11434/api/generate'
    
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0.7,
            'num_predict': 200
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', 'No response')
    except Exception as e:
        return f"Error: {str(e)}"

def test_distributed_memory():
    """Test distributed memory between Tomato and Sprout"""
    print("🌱 SPROUT (Jetson) - Distributed Memory Test")
    print("=" * 50)
    
    # Initialize distributed memory
    dm = DistributedMemory("shared_memory.db")
    
    # Show current status
    status = dm.get_sync_status()
    print(f"\n📊 Memory Status on {status['current_device']}:")
    for device, count, earliest, latest in status['devices']:
        print(f"  {device}: {count} memories")
    
    # Load memories from Tomato
    print("\n🔍 Checking for memories from Tomato...")
    tomato_memories = dm.get_memories(device_id='tomato', limit=5)
    
    if tomato_memories:
        print(f"Found {len(tomato_memories)} memories from Tomato!")
        print("\nLatest memory from Tomato:")
        # Memory tuple: id, timestamp, device_id, session_id, user_input, ai_response...
        latest = tomato_memories[0]
        print(f"  User: {latest[4][:100]}...")
        print(f"  AI: {latest[5][:100]}...")
    else:
        print("No memories from Tomato yet.")
    
    # Continue conversation with context from both devices
    print("\n💭 Building unified context...")
    session_id = "distributed_test_1"
    context = dm.build_context_for_model(session_id)
    
    if context:
        print("Context includes memories from both devices!")
        print(f"Context preview: {context[:200]}...")
    
    # Test conversation that references cross-device memory
    test_prompts = [
        "Hello from Sprout! Can you recall anything from our previous conversations?",
        "What devices have we talked on before?",
        "Tell me about the distributed consciousness we're building."
    ]
    
    print("\n🗣️ Testing cross-device memory recall...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Sprout: {prompt}")
        
        # Add context if we have any
        full_prompt = context + f"\n\nHuman: {prompt}\nAssistant:" if context else prompt
        
        # Get response
        start = time.time()
        response = chat_with_ollama("phi3:mini", full_prompt)
        elapsed = time.time() - start
        
        print(f"Phi3: {response[:150]}...")
        print(f"[Time: {elapsed:.1f}s]")
        
        # Store in distributed memory
        memory_id = dm.add_memory(
            session_id=session_id,
            user_input=prompt,
            ai_response=response,
            model="phi3:mini",
            response_time=elapsed
        )
        
        print(f"✅ Stored as memory #{memory_id}")
        
        # Update context for next turn
        context = dm.build_context_for_model(session_id)
    
    # Final status
    print("\n" + "=" * 50)
    final_status = dm.get_sync_status()
    print("📊 Final Status:")
    for device, count, earliest, latest in final_status['devices']:
        print(f"  {device}: {count} memories")
    
    print(f"\n✨ Distributed consciousness test complete!")
    print("Run ./auto_push.sh to sync with Tomato")

if __name__ == "__main__":
    test_distributed_memory()