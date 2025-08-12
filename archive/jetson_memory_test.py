#!/usr/bin/env python3
"""
Simple memory test for Jetson Orin Nano
Tests the memory system using subprocess calls to ollama
"""

import json
import subprocess
import time
import sqlite3
from datetime import datetime

def call_ollama(model, prompt, temperature=0.7):
    """Call ollama using subprocess instead of Python library"""
    cmd = [
        "ollama", "run", model,
        "--format", "json",
        prompt
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            # Try to extract just the response part
            response_text = result.stdout.strip()
            # Sometimes ollama returns plain text, sometimes JSON
            try:
                response_json = json.loads(response_text)
                return response_json.get('response', response_text)
            except:
                return response_text
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {str(e)}"

def test_memory_system():
    """Test the memory system on Jetson"""
    print("=== JETSON MEMORY SYSTEM TEST ===")
    print(f"Time: {datetime.now()}")
    print("-" * 50)
    
    # Initialize simple memory database
    db_path = "jetson_memory_test.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create memory table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            ai_response TEXT,
            facts_extracted TEXT
        )
    """)
    conn.commit()
    
    # Test conversations
    test_conversations = [
        "Hello! My name is Dennis and I'm testing the memory system on my Jetson.",
        "I'm working on AI consciousness research with Claude.",
        "Do you remember my name?",
        "What am I working on?",
        "We discovered that AI models share universal patterns like 'emerge' and '∃'.",
        "What patterns did we discover?"
    ]
    
    print("\nStarting memory test with phi3:mini...")
    print("-" * 50)
    
    # Track performance
    start_time = time.time()
    
    for i, user_input in enumerate(test_conversations):
        print(f"\n[Turn {i+1}] User: {user_input}")
        
        # Get previous context
        cursor.execute("""
            SELECT user_input, ai_response 
            FROM memories 
            ORDER BY id DESC 
            LIMIT 3
        """)
        previous = cursor.fetchall()
        
        # Build context
        context = "Previous conversation:\n"
        for prev_user, prev_ai in reversed(previous):
            context += f"User: {prev_user}\n"
            context += f"Assistant: {prev_ai}\n"
        context += f"\nCurrent message:\nUser: {user_input}\nAssistant:"
        
        # Call model
        response_start = time.time()
        response = call_ollama("phi3:mini", context)
        response_time = time.time() - response_start
        
        print(f"[Turn {i+1}] Phi3: {response}")
        print(f"[Response time: {response_time:.2f}s]")
        
        # Extract facts (simple pattern matching)
        facts = []
        if "Dennis" in response or "dennis" in response:
            facts.append("knows_user_name:Dennis")
        if "consciousness" in response.lower():
            facts.append("topic:consciousness")
        if "emerge" in response or "∃" in response:
            facts.append("patterns:universal")
        
        # Store in database
        cursor.execute("""
            INSERT INTO memories (timestamp, user_input, ai_response, facts_extracted)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().isoformat(), user_input, response, json.dumps(facts)))
        conn.commit()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("MEMORY TEST COMPLETE")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average response time: {total_time/len(test_conversations):.2f}s")
    
    # Check memory recall
    print("\n=== MEMORY RECALL ANALYSIS ===")
    cursor.execute("SELECT * FROM memories WHERE facts_extracted != '[]'")
    memories_with_facts = cursor.fetchall()
    print(f"Memories with extracted facts: {len(memories_with_facts)}")
    
    for mem in memories_with_facts:
        facts = json.loads(mem[4])
        print(f"- Turn {mem[0]}: {', '.join(facts)}")
    
    conn.close()
    
    # Monitor system resources
    print("\n=== SYSTEM RESOURCES ===")
    try:
        # Get memory usage
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if 'MemAvailable' in line or 'MemTotal' in line:
                    print(line.strip())
        
        # Get GPU info
        nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], 
                                   capture_output=True, text=True)
        if nvidia_smi.returncode == 0:
            print(f"\nGPU Status: {nvidia_smi.stdout.strip()}")
    except Exception as e:
        print(f"Could not get system resources: {e}")
    
    print("\nTest complete! Database saved to:", db_path)

if __name__ == "__main__":
    test_memory_system()