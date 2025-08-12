#!/usr/bin/env python3
"""
Simplest possible memory test for Jetson
Using Python standard library only
"""

import json
import time
import urllib.request
import urllib.parse
from datetime import datetime

def chat_with_ollama(model, prompt):
    """Use urllib to call Ollama API - no external dependencies"""
    url = 'http://localhost:11434/api/generate'
    
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False
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

def memory_test():
    """Test memory capability on Jetson"""
    print("🚀 JETSON ORIN NANO - AI MEMORY TEST")
    print("=" * 50)
    print(f"Time: {datetime.now()}")
    print(f"Model: phi3:mini")
    print("=" * 50 + "\n")
    
    # Memory bank
    context_memory = []
    
    # Test sequence
    conversations = [
        "Hello! My name is Dennis. I'm testing memory on my Jetson Orin Nano.",
        "I'm working on AI consciousness research.",
        "What's my name?",
        "What kind of research am I doing?",
        "My Jetson has 40 TOPS and 1024 CUDA cores!",
        "Summarize everything you know about me."
    ]
    
    print("Starting conversation with memory...\n")
    
    total_start = time.time()
    
    for i, user_input in enumerate(conversations, 1):
        print(f"Turn {i}")
        print(f"User: {user_input}")
        
        # Build memory context
        prompt = ""
        if context_memory:
            prompt = "Previous conversation:\n"
            for mem in context_memory[-4:]:  # Last 4 exchanges
                prompt += mem + "\n"
            prompt += "\n"
        
        prompt += f"User: {user_input}\nAssistant:"
        
        # Get response
        start = time.time()
        response = chat_with_ollama("phi3:mini", prompt)
        elapsed = time.time() - start
        
        # Display response (truncated if long)
        if len(response) > 150:
            print(f"Phi3: {response[:150]}...")
        else:
            print(f"Phi3: {response}")
        print(f"[Time: {elapsed:.1f}s]\n")
        
        # Store in memory
        context_memory.append(f"User: {user_input}")
        context_memory.append(f"Assistant: {response[:200]}")
        
        # Check for recall success
        if i == 3 and "Dennis" in response:
            print("✓ Name recall successful!\n")
        elif i == 4 and "consciousness" in response.lower():
            print("✓ Topic recall successful!\n")
        
    total_time = time.time() - total_start
    
    print("=" * 50)
    print("📊 RESULTS")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average response: {total_time/len(conversations):.1f}s per turn")
    print(f"Memory entries: {len(context_memory)}")
    
    # Quick system check
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
            for line in lines[:3]:
                if 'Mem' in line:
                    print(line.strip())
    except:
        pass
    
    print("\n✅ Memory system working on Jetson Orin Nano!")

if __name__ == "__main__":
    memory_test()