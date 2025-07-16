#!/usr/bin/env python3
"""
Enhanced Phi3 Memory System with Semantic Search and Better Fact Extraction
"""

import json
import sqlite3
import time
import hashlib
import subprocess
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

class EnhancedPhi3Memory:
    """Enhanced memory system with semantic search and better fact extraction"""
    
    def __init__(self, db_path="/home/dp/ai-workspace/ai-agents/phi3_memory_enhanced.db", 
                 window_size=10, max_context_tokens=2000):
        self.db_path = db_path
        self.window_size = window_size
        self.max_context_tokens = max_context_tokens
        self.model_name = "phi3:mini"
        
        # Initialize database with enhanced schema
        self._init_enhanced_db()
        
        # Memory types
        self.episodic_memory = []    # Specific events
        self.semantic_memory = {}    # General knowledge
        self.working_memory = []     # Current context
        
        # Fact patterns for better extraction
        self.fact_patterns = {
            'identity': [
                r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+)",
                r"(?:this is)\s+([A-Z][a-z]+)\s+(?:speaking|here)"
            ],
            'profession': [
                r"i(?:'m| am)?\s+(?:a|an)\s+([\w\s]+)(?:\.|,|$)",
                r"i work as\s+(?:a|an)?\s*([\w\s]+)",
                r"my job is\s+([\w\s]+)"
            ],
            'preference': [
                r"i (?:like|love|enjoy|prefer)\s+([\w\s]+)",
                r"my favorite\s+([\w\s]+)\s+is\s+([\w\s]+)"
            ],
            'skill': [
                r"i (?:can|know how to)\s+([\w\s]+)",
                r"i'm (?:good at|skilled in)\s+([\w\s]+)"
            ],
            'location': [
                r"i live in\s+([\w\s]+)",
                r"i'm from\s+([\w\s]+)"
            ]
        }
    
    def _init_enhanced_db(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT,
                importance_score REAL DEFAULT 0.5,
                metadata TEXT
            )
        ''')
        
        # Enhanced facts table with categories
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                frequency INTEGER DEFAULT 1,
                last_mentioned DATETIME DEFAULT CURRENT_TIMESTAMP,
                source_message_id INTEGER,
                FOREIGN KEY (source_message_id) REFERENCES conversations(id)
            )
        ''')
        
        # Semantic memory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT NOT NULL,
                definition TEXT,
                related_concepts TEXT,
                importance REAL DEFAULT 0.5,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
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
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_time ON conversations(session_id, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fact_session ON facts(session_id, fact_type)')
        
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
    
    def extract_structured_facts(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract structured facts with confidence scores"""
        facts = defaultdict(list)
        text_lower = text.lower()
        
        for fact_type, patterns in self.fact_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    if match.groups():
                        fact_value = match.group(1).strip()
                        # Calculate confidence based on context
                        confidence = 0.9 if len(fact_value.split()) < 5 else 0.7
                        facts[fact_type].append((fact_value, confidence))
        
        # Also extract custom facts from direct statements
        if "remember" in text_lower or "don't forget" in text_lower:
            important_part = text_lower.split("remember")[-1] if "remember" in text_lower else text_lower.split("don't forget")[-1]
            facts['important'].append((important_part.strip(), 0.95))
        
        return dict(facts)
    
    def update_semantic_memory(self, concept: str, context: str):
        """Update semantic memory with new concepts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if concept exists
        cursor.execute(
            "SELECT id, importance FROM semantic_memory WHERE concept = ?",
            (concept,)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update importance and last accessed
            new_importance = min(1.0, existing[1] + 0.1)
            cursor.execute(
                "UPDATE semantic_memory SET importance = ?, last_accessed = CURRENT_TIMESTAMP WHERE id = ?",
                (new_importance, existing[0])
            )
        else:
            # Add new concept
            cursor.execute(
                "INSERT INTO semantic_memory (concept, definition) VALUES (?, ?)",
                (concept, context)
            )
        
        conn.commit()
        conn.close()
    
    def get_relevant_facts(self, session_id: str, query: str) -> List[Dict]:
        """Get facts relevant to the current query"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple relevance: get all facts for session, prioritize by frequency and recency
        cursor.execute('''
            SELECT fact_type, fact_key, fact_value, confidence, frequency
            FROM facts
            WHERE session_id = ?
            ORDER BY frequency DESC, last_mentioned DESC
            LIMIT 10
        ''', (session_id,))
        
        facts = []
        for row in cursor.fetchall():
            facts.append({
                'type': row[0],
                'key': row[1],
                'value': row[2],
                'confidence': row[3],
                'frequency': row[4]
            })
        
        conn.close()
        return facts
    
    def build_enhanced_context(self, session_id: str, current_query: str) -> str:
        """Build context with relevance scoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get relevant facts
        facts = self.get_relevant_facts(session_id, current_query)
        
        # Get recent conversation with importance scores
        cursor.execute('''
            SELECT role, content, importance_score
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, self.window_size * 2))
        
        recent_exchanges = cursor.fetchall()
        recent_exchanges.reverse()
        
        # Build context sections
        context_parts = []
        
        # Add personality/identity facts first
        identity_facts = [f for f in facts if f['type'] in ['identity', 'profession']]
        if identity_facts:
            context_parts.append("About you:")
            for fact in identity_facts:
                context_parts.append(f"- You are {fact['value']}")
        
        # Add preferences and skills
        pref_facts = [f for f in facts if f['type'] in ['preference', 'skill']]
        if pref_facts:
            context_parts.append("\nYour interests:")
            for fact in pref_facts:
                context_parts.append(f"- You {fact['key']} {fact['value']}")
        
        # Add important memories
        important_facts = [f for f in facts if f['type'] == 'important']
        if important_facts:
            context_parts.append("\nImportant to remember:")
            for fact in important_facts:
                context_parts.append(f"- {fact['value']}")
        
        # Add recent conversation
        if recent_exchanges:
            context_parts.append("\nRecent conversation:")
            for role, content, importance in recent_exchanges:
                # Only include if importance > 0.3
                if importance > 0.3:
                    if role == "user":
                        context_parts.append(f"Human: {content}")
                    else:
                        context_parts.append(f"Assistant: {content}")
        
        conn.close()
        
        context = "\n".join(context_parts)
        
        # Smart truncation - keep identity facts, truncate conversation
        if len(context) > self.max_context_tokens * 4:
            # Split into sections
            sections = context.split("\n\n")
            # Keep first 2 sections (identity/interests) and truncate conversation
            if len(sections) > 2:
                kept_sections = sections[:2]
                conversation = "\n\n".join(sections[2:])
                # Truncate conversation
                conversation = conversation[-(self.max_context_tokens * 2):]
                context = "\n\n".join(kept_sections) + "\n\n" + conversation
        
        return context
    
    def calculate_importance(self, message: str, role: str) -> float:
        """Calculate importance score for a message"""
        importance = 0.5  # Base score
        
        # Questions are important
        if "?" in message:
            importance += 0.2
        
        # Messages with facts are important
        if any(pattern in message.lower() for pattern in ["my name", "i am", "i like", "remember"]):
            importance += 0.3
        
        # User messages slightly more important
        if role == "user":
            importance += 0.1
        
        return min(1.0, importance)
    
    def add_enhanced_exchange(self, session_id: str, user_input: str, ai_response: str):
        """Add exchange with enhanced metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate importance scores
        user_importance = self.calculate_importance(user_input, "user")
        ai_importance = self.calculate_importance(ai_response, "assistant")
        
        # Add messages with importance
        cursor.execute(
            "INSERT INTO conversations (session_id, role, content, importance_score) VALUES (?, ?, ?, ?)",
            (session_id, "user", user_input, user_importance)
        )
        user_msg_id = cursor.lastrowid
        
        cursor.execute(
            "INSERT INTO conversations (session_id, role, content, importance_score) VALUES (?, ?, ?, ?)",
            (session_id, "assistant", ai_response, ai_importance)
        )
        
        # Extract and store facts
        user_facts = self.extract_structured_facts(user_input)
        for fact_type, fact_list in user_facts.items():
            for fact_value, confidence in fact_list:
                # Check if fact exists
                cursor.execute(
                    "SELECT id, frequency FROM facts WHERE session_id = ? AND fact_type = ? AND fact_value = ?",
                    (session_id, fact_type, fact_value)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update frequency
                    cursor.execute(
                        "UPDATE facts SET frequency = ?, last_mentioned = CURRENT_TIMESTAMP WHERE id = ?",
                        (existing[1] + 1, existing[0])
                    )
                else:
                    # Insert new fact
                    cursor.execute(
                        "INSERT INTO facts (session_id, fact_type, fact_key, fact_value, confidence, source_message_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (session_id, fact_type, fact_type, fact_value, confidence, user_msg_id)
                    )
        
        conn.commit()
        conn.close()
    
    def query_with_enhanced_memory(self, session_id: str, user_input: str, 
                                  temperature: float = 0.7) -> str:
        """Query with enhanced memory system"""
        # Build enhanced context
        context = self.build_enhanced_context(session_id, user_input)
        
        # Build prompt
        if context:
            full_prompt = f"{context}\n\nHuman: {user_input}\nAssistant:"
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
                
                # Store enhanced exchange
                self.add_enhanced_exchange(session_id, user_input, ai_response)
                
                return ai_response
        except Exception as e:
            print(f"Error: {e}")
            return "I encountered an error processing your request."
    
    def get_memory_stats(self, session_id: str) -> Dict:
        """Get detailed memory statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Message stats
        cursor.execute(
            "SELECT COUNT(*), AVG(importance_score) FROM conversations WHERE session_id = ?",
            (session_id,)
        )
        msg_count, avg_importance = cursor.fetchone()
        stats['message_count'] = msg_count or 0
        stats['avg_importance'] = avg_importance or 0
        
        # Fact stats by type
        cursor.execute(
            "SELECT fact_type, COUNT(*), AVG(confidence) FROM facts WHERE session_id = ? GROUP BY fact_type",
            (session_id,)
        )
        stats['facts_by_type'] = {}
        for fact_type, count, avg_conf in cursor.fetchall():
            stats['facts_by_type'][fact_type] = {
                'count': count,
                'avg_confidence': avg_conf
            }
        
        conn.close()
        return stats


def demo_enhanced_memory():
    """Demonstrate enhanced memory system"""
    print("ENHANCED PHI3 MEMORY SYSTEM DEMO")
    print("=" * 50)
    
    memory = EnhancedPhi3Memory()
    session_id = memory.create_session()
    
    print(f"\nSession: {session_id}")
    print("\n--- Enhanced Conversation with Better Memory ---\n")
    
    # More complex conversation
    exchanges = [
        "Hi! I'm Bob, a software engineer specializing in AI systems.",
        "I love hiking and photography. My favorite programming language is Python.",
        "Remember that I have a important meeting tomorrow at 3pm.",
        "What did I tell you about my meeting?",
        "What are my hobbies?",
        "What's my profession again?",
    ]
    
    for user_input in exchanges:
        print(f"Human: {user_input}")
        response = memory.query_with_enhanced_memory(session_id, user_input, temperature=0.7)
        print(f"Phi3: {response}\n")
        time.sleep(2)
    
    # Show detailed stats
    print("\n--- Memory Statistics ---")
    stats = memory.get_memory_stats(session_id)
    print(f"Total messages: {stats['message_count']}")
    print(f"Average importance: {stats['avg_importance']:.2f}")
    print(f"Facts by type:")
    for fact_type, info in stats['facts_by_type'].items():
        print(f"  {fact_type}: {info['count']} facts (confidence: {info['avg_confidence']:.2f})")


if __name__ == "__main__":
    demo_enhanced_memory()