#!/usr/bin/env python3
"""
Distributed Memory System for AI DNA Discovery
Works across Laptop (Tomato) and Jetson (Sprout)
"""

import sqlite3
import json
import os
from datetime import datetime
import platform
import socket

class DistributedMemory:
    def __init__(self, db_path="shared_memory.db"):
        self.db_path = db_path
        self.device_id = self._get_device_id()
        self._init_db()
        
    def _get_device_id(self):
        """Identify device - Tomato (laptop) or Sprout (jetson)"""
        hostname = socket.gethostname().lower()
        if 'jetson' in hostname or hostname == 'sprout':
            return 'sprout'
        elif hostname == 'tomato':
            return 'tomato'
        else:
            return hostname
    
    def _init_db(self):
        """Initialize unified database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main memories table - expanded from Jetson's simple version
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                device_id TEXT NOT NULL,
                session_id TEXT,
                user_input TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                model_used TEXT,
                response_time REAL,
                facts_extracted TEXT,
                importance_score REAL DEFAULT 0.5
            )
        ''')
        
        # Facts table for structured knowledge
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                device_id TEXT NOT NULL,
                session_id TEXT,
                fact_type TEXT,  -- 'identity', 'preference', 'context'
                fact_value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_memory_id INTEGER,
                FOREIGN KEY (source_memory_id) REFERENCES memories(id)
            )
        ''')
        
        # Context tokens for state preservation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                device_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_compressed TEXT,  -- Base64 encoded compressed tokens
                token_count INTEGER,
                compression_ratio REAL
            )
        ''')
        
        # Sync metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                device_id TEXT NOT NULL,
                action TEXT,  -- 'push', 'pull', 'merge'
                records_affected INTEGER,
                status TEXT  -- 'success', 'conflict', 'error'
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_device ON memories(device_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id)')
        
        conn.commit()
        conn.close()
        
        print(f"📊 Distributed memory initialized on {self.device_id}")
    
    def add_memory(self, session_id, user_input, ai_response, model="phi3:mini", 
                   response_time=None, facts=None):
        """Add a memory entry from current device"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Insert memory
        cursor.execute('''
            INSERT INTO memories (timestamp, device_id, session_id, user_input, 
                                ai_response, model_used, response_time, facts_extracted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, self.device_id, session_id, user_input, ai_response, 
              model, response_time, json.dumps(facts) if facts else None))
        
        memory_id = cursor.lastrowid
        
        # Extract and store structured facts
        if facts:
            for fact_type, fact_list in facts.items():
                for fact_value, confidence in fact_list:
                    cursor.execute('''
                        INSERT INTO facts (timestamp, device_id, session_id, 
                                         fact_type, fact_value, confidence, source_memory_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (timestamp, self.device_id, session_id, fact_type, 
                          fact_value, confidence, memory_id))
        
        conn.commit()
        conn.close()
        
        return memory_id
    
    def get_memories(self, session_id=None, device_id=None, limit=10):
        """Retrieve memories with optional filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM memories WHERE 1=1'
        params = []
        
        if session_id:
            query += ' AND session_id = ?'
            params.append(session_id)
        
        if device_id:
            query += ' AND device_id = ?'
            params.append(device_id)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        memories = cursor.fetchall()
        
        conn.close()
        return memories
    
    def get_facts(self, session_id=None, fact_type=None):
        """Retrieve structured facts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM facts WHERE confidence > 0.5'
        params = []
        
        if session_id:
            query += ' AND session_id = ?'
            params.append(session_id)
        
        if fact_type:
            query += ' AND fact_type = ?'
            params.append(fact_type)
        
        query += ' ORDER BY confidence DESC, timestamp DESC'
        
        cursor.execute(query, params)
        facts = cursor.fetchall()
        
        conn.close()
        return facts
    
    def migrate_from_simple_db(self, simple_db_path):
        """Migrate from Jetson's simple memory database"""
        if not os.path.exists(simple_db_path):
            return 0
        
        print(f"🔄 Migrating from {simple_db_path}...")
        
        # Connect to both databases
        simple_conn = sqlite3.connect(simple_db_path)
        simple_cursor = simple_conn.cursor()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Read from simple database
        simple_cursor.execute('SELECT * FROM memories ORDER BY timestamp')
        simple_memories = simple_cursor.fetchall()
        
        # Migrate each memory
        migrated = 0
        for memory in simple_memories:
            # Simple schema: id, timestamp, user_input, ai_response, facts_extracted
            _, timestamp, user_input, ai_response, facts_json = memory
            
            # Check if already migrated
            cursor.execute('''
                SELECT COUNT(*) FROM memories 
                WHERE timestamp = ? AND user_input = ?
            ''', (timestamp, user_input))
            
            if cursor.fetchone()[0] == 0:
                # Insert into new schema
                cursor.execute('''
                    INSERT INTO memories (timestamp, device_id, session_id, user_input, 
                                        ai_response, model_used, facts_extracted)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, 'sprout', 'jetson_test_1', user_input, 
                      ai_response, 'phi3:mini', facts_json))
                migrated += 1
        
        conn.commit()
        simple_conn.close()
        conn.close()
        
        print(f"✅ Migrated {migrated} memories from Jetson")
        return migrated
    
    def get_sync_status(self):
        """Get summary of memory distribution"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count memories by device
        cursor.execute('''
            SELECT device_id, COUNT(*) as count, 
                   MIN(timestamp) as earliest,
                   MAX(timestamp) as latest
            FROM memories
            GROUP BY device_id
        ''')
        
        device_stats = cursor.fetchall()
        
        # Total facts
        cursor.execute('SELECT COUNT(*) FROM facts')
        total_facts = cursor.fetchone()[0]
        
        # Database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        db_size = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'devices': device_stats,
            'total_facts': total_facts,
            'db_size_bytes': db_size,
            'current_device': self.device_id
        }
    
    def build_context_for_model(self, session_id, max_tokens=2000):
        """Build context from memories for model input"""
        memories = self.get_memories(session_id, limit=20)
        facts = self.get_facts(session_id)
        
        context_parts = []
        
        # Add important facts first
        if facts:
            context_parts.append("Known facts:")
            for fact in facts[:5]:  # Top 5 facts
                # fact schema: id, timestamp, device_id, session_id, fact_type, fact_value, confidence, source_id
                fact_type = fact[4]
                fact_value = fact[5]
                confidence = fact[6]
                if confidence > 0.7:
                    context_parts.append(f"- {fact_type}: {fact_value}")
        
        # Add recent conversation
        if memories:
            context_parts.append("\nRecent conversation:")
            for memory in reversed(memories[-10:]):  # Last 10, in chronological order
                # memory schema has many fields, we want user_input (4) and ai_response (5)
                user_input = memory[4]
                ai_response = memory[5]
                context_parts.append(f"Human: {user_input}")
                context_parts.append(f"Assistant: {ai_response[:200]}...")  # Truncate long responses
        
        return "\n".join(context_parts)

if __name__ == "__main__":
    # Test the distributed memory system
    print("🧠 Testing Distributed Memory System")
    print("=" * 50)
    
    dm = DistributedMemory()
    
    # Check for Jetson database to migrate
    if os.path.exists("jetson_memory_test.db"):
        dm.migrate_from_simple_db("jetson_memory_test.db")
    
    # Show sync status
    status = dm.get_sync_status()
    print(f"\n📊 Sync Status:")
    print(f"Current device: {status['current_device']}")
    print(f"Total facts: {status['total_facts']}")
    print(f"Database size: {status['db_size_bytes']/1024:.1f} KB")
    
    if status['devices']:
        print("\nMemories by device:")
        for device, count, earliest, latest in status['devices']:
            print(f"  {device}: {count} memories ({earliest[:10]} to {latest[:10]})")
    
    print("\n✅ Distributed memory system ready!")