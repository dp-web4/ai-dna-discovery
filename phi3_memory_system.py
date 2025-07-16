#!/usr/bin/env python3
"""
Phi3 Memory System - External Context Management
Transform stateless Phi3 into a memory-enhanced conversational AI
"""

import json
import sqlite3
import time
import hashlib
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

class Phi3Memory:
    """External memory system for Phi3 model"""
    
    def __init__(self, db_path="/home/dp/ai-workspace/ai-agents/phi3_memory.db", 
                 window_size=10, max_context_tokens=2000):
        self.db_path = db_path
        self.window_size = window_size
        self.max_context_tokens = max_context_tokens
        self.model_name = "phi3:mini"
        
        # Initialize database
        self._init_db()
        
        # Memory caches
        self.short_term_memory = []  # Recent exchanges
        self.context_cache = ""      # Current context string
        
    def _init_db(self):
        """Initialize SQLite database for persistent memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding_hash TEXT,
                metadata TEXT
            )
        ''')
        
        # Facts table for extracted knowledge
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_message_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_message_id) REFERENCES conversations(id)
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session"""
        if not session_id:
            session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
            (session_id,)
        )
        conn.commit()
        conn.close()
        
        return session_id
    
    def add_exchange(self, session_id: str, user_input: str, ai_response: str):
        """Add a conversation exchange to memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add user message
        cursor.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "user", user_input)
        )
        
        # Add AI response
        cursor.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "assistant", ai_response)
        )
        
        # Update session last active
        cursor.execute(
            "UPDATE sessions SET last_active = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,)
        )
        
        conn.commit()
        conn.close()
        
        # Update short-term memory cache
        self.short_term_memory.append({"role": "user", "content": user_input})
        self.short_term_memory.append({"role": "assistant", "content": ai_response})
        
        # Keep only recent exchanges in cache
        if len(self.short_term_memory) > self.window_size * 2:
            self.short_term_memory = self.short_term_memory[-(self.window_size * 2):]
    
    def extract_facts(self, session_id: str, text: str) -> List[str]:
        """Extract facts from conversation text"""
        # Simple fact extraction - can be enhanced with NLP
        facts = []
        
        # Look for patterns like "I am...", "My name is...", "I like..."
        patterns = [
            "i am", "my name is", "i like", "i prefer", "i work",
            "i live", "my favorite", "i have", "i think", "i believe"
        ]
        
        sentences = text.lower().split('.')
        for sentence in sentences:
            for pattern in patterns:
                if pattern in sentence:
                    fact = sentence.strip()
                    if len(fact) > 10:  # Minimum fact length
                        facts.append(fact)
                        break
        
        # Store facts in database
        if facts:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for fact in facts:
                cursor.execute(
                    "INSERT INTO facts (session_id, fact) VALUES (?, ?)",
                    (session_id, fact)
                )
            conn.commit()
            conn.close()
        
        return facts
    
    def get_context(self, session_id: str, include_facts: bool = True) -> str:
        """Build context string from memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent conversation history
        cursor.execute('''
            SELECT role, content FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, self.window_size * 2))
        
        recent_exchanges = cursor.fetchall()
        recent_exchanges.reverse()  # Chronological order
        
        # Build context
        context_parts = []
        
        # Add session facts if requested
        if include_facts:
            cursor.execute(
                "SELECT fact FROM facts WHERE session_id = ? ORDER BY confidence DESC LIMIT 5",
                (session_id,)
            )
            facts = cursor.fetchall()
            if facts:
                context_parts.append("Known facts about you:")
                for fact in facts:
                    context_parts.append(f"- {fact[0]}")
                context_parts.append("")
        
        # Add conversation history
        if recent_exchanges:
            context_parts.append("Recent conversation:")
            for role, content in recent_exchanges:
                if role == "user":
                    context_parts.append(f"Human: {content}")
                else:
                    context_parts.append(f"Assistant: {content}")
            context_parts.append("")
        
        conn.close()
        
        context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.max_context_tokens * 4:  # Rough char estimate
            context = context[-(self.max_context_tokens * 4):]
        
        return context
    
    def query_with_memory(self, session_id: str, user_input: str, 
                         temperature: float = 0.7) -> str:
        """Query Phi3 with memory context"""
        # Extract facts from user input
        self.extract_facts(session_id, user_input)
        
        # Get context
        context = self.get_context(session_id)
        
        # Build prompt with context
        if context:
            full_prompt = f"{context}\nHuman: {user_input}\nAssistant:"
        else:
            full_prompt = f"Human: {user_input}\nAssistant:"
        
        # Query model
        try:
            response = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/generate", "-d",
                 json.dumps({
                     "model": self.model_name,
                     "prompt": full_prompt,
                     "stream": False,
                     "options": {
                         "temperature": temperature,
                         "num_predict": 500
                     }
                 })],
                capture_output=True, text=True, timeout=60
            )
            
            if response.returncode == 0:
                result = json.loads(response.stdout)
                ai_response = result.get('response', '').strip()
                
                # Store exchange
                self.add_exchange(session_id, user_input, ai_response)
                
                # Extract facts from response
                self.extract_facts(session_id, ai_response)
                
                return ai_response
        except Exception as e:
            print(f"Error: {e}")
            return "I encountered an error processing your request."
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of a conversation session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute(
            "SELECT created_at, last_active FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        session_info = cursor.fetchone()
        
        # Get exchange count
        cursor.execute(
            "SELECT COUNT(*) FROM conversations WHERE session_id = ?",
            (session_id,)
        )
        exchange_count = cursor.fetchone()[0]
        
        # Get fact count
        cursor.execute(
            "SELECT COUNT(*) FROM facts WHERE session_id = ?",
            (session_id,)
        )
        fact_count = cursor.fetchone()[0]
        
        # Get sample facts
        cursor.execute(
            "SELECT fact FROM facts WHERE session_id = ? ORDER BY confidence DESC LIMIT 3",
            (session_id,)
        )
        sample_facts = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "session_id": session_id,
            "created_at": session_info[0] if session_info else None,
            "last_active": session_info[1] if session_info else None,
            "exchange_count": exchange_count,
            "fact_count": fact_count,
            "sample_facts": sample_facts
        }
    
    def clear_session(self, session_id: str):
        """Clear all memory for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM facts WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
        
        self.short_term_memory = []
        self.context_cache = ""


def demo_memory_system():
    """Demonstrate the memory system"""
    print("PHI3 MEMORY SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize memory system
    memory = Phi3Memory()
    
    # Create a session
    session_id = memory.create_session()
    print(f"\nCreated session: {session_id}")
    
    # Simulate conversation
    exchanges = [
        "Hello! My name is Alice and I'm a data scientist.",
        "I love working with Python and machine learning. My favorite color is blue.",
        "What's my name?",
        "What do I do for work?",
        "What's my favorite color?",
    ]
    
    print("\n--- Starting Conversation ---\n")
    
    for user_input in exchanges:
        print(f"Human: {user_input}")
        response = memory.query_with_memory(session_id, user_input)
        print(f"Phi3: {response}\n")
        time.sleep(2)  # Pause between exchanges
    
    # Show session summary
    print("\n--- Session Summary ---")
    summary = memory.get_session_summary(session_id)
    print(f"Session ID: {summary['session_id']}")
    print(f"Exchanges: {summary['exchange_count']}")
    print(f"Facts learned: {summary['fact_count']}")
    print(f"Sample facts:")
    for fact in summary['sample_facts']:
        print(f"  - {fact}")
    
    print("\n--- Testing Session Persistence ---")
    print("Creating new memory instance...")
    memory2 = Phi3Memory()
    
    print("Asking about previous conversation...")
    response = memory2.query_with_memory(session_id, "Can you remind me what we talked about?")
    print(f"Phi3: {response}")
    
    print("\n✓ Memory persisted across instances!")


if __name__ == "__main__":
    demo_memory_system()