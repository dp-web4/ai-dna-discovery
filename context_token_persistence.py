#!/usr/bin/env python3
"""
Context Token Persistence System
Using Ollama's context tokens as portable KV-cache
"""

import json
import sqlite3
import base64
import zlib
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class ContextTokenPersistence:
    """Manage context tokens as portable model state"""
    
    def __init__(self, db_path="/home/dp/ai-workspace/ai-agents/context_tokens.db"):
        self.db_path = db_path
        self.api_base = "http://localhost:11434"
        self._init_db()
    
    def _init_db(self):
        """Initialize context token database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Context storage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                context_data TEXT NOT NULL,
                context_size INTEGER,
                compressed_size INTEGER,
                prompt TEXT,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Session checkpoints
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                context_tokens_id INTEGER,
                checkpoint_name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (context_tokens_id) REFERENCES context_tokens(id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_model ON context_tokens(session_id, model_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoint_session ON session_checkpoints(session_id)')
        
        conn.commit()
        conn.close()
    
    def compress_context(self, context_tokens: List[int]) -> str:
        """Compress context tokens for efficient storage"""
        # Convert to bytes
        context_bytes = json.dumps(context_tokens).encode('utf-8')
        
        # Compress
        compressed = zlib.compress(context_bytes, level=9)
        
        # Base64 encode for storage
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        return encoded
    
    def decompress_context(self, encoded_context: str) -> List[int]:
        """Decompress stored context tokens"""
        # Base64 decode
        compressed = base64.b64decode(encoded_context.encode('utf-8'))
        
        # Decompress
        context_bytes = zlib.decompress(compressed)
        
        # Convert back to list
        context_tokens = json.loads(context_bytes.decode('utf-8'))
        
        return context_tokens
    
    def save_context(self, session_id: str, model_name: str, 
                    context_tokens: List[int], prompt: str, response: str) -> int:
        """Save context tokens to database"""
        # Compress context
        compressed_context = self.compress_context(context_tokens)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO context_tokens 
            (session_id, model_name, context_data, context_size, compressed_size, prompt, response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            model_name,
            compressed_context,
            len(context_tokens),
            len(compressed_context),
            prompt,
            response
        ))
        
        context_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return context_id
    
    def load_context(self, session_id: str, model_name: str) -> Optional[Tuple[List[int], Dict]]:
        """Load most recent context for session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT context_data, prompt, response, context_size, compressed_size, timestamp
            FROM context_tokens
            WHERE session_id = ? AND model_name = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (session_id, model_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            context_tokens = self.decompress_context(result[0])
            metadata = {
                'last_prompt': result[1],
                'last_response': result[2],
                'original_size': result[3],
                'compressed_size': result[4],
                'timestamp': result[5]
            }
            return context_tokens, metadata
        
        return None
    
    def create_checkpoint(self, session_id: str, model_name: str, 
                         checkpoint_name: str) -> str:
        """Create a named checkpoint of current context"""
        import hashlib
        
        # Get latest context
        result = self.load_context(session_id, model_name)
        if not result:
            raise ValueError("No context found for session")
        
        # Generate checkpoint ID
        checkpoint_id = hashlib.sha256(
            f"{session_id}:{checkpoint_name}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get latest context ID
        cursor.execute('''
            SELECT id FROM context_tokens
            WHERE session_id = ? AND model_name = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (session_id, model_name))
        
        context_id = cursor.fetchone()[0]
        
        # Create checkpoint
        cursor.execute('''
            INSERT INTO session_checkpoints
            (checkpoint_id, session_id, model_name, context_tokens_id, checkpoint_name)
            VALUES (?, ?, ?, ?, ?)
        ''', (checkpoint_id, session_id, model_name, context_id, checkpoint_name))
        
        conn.commit()
        conn.close()
        
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> Tuple[List[int], Dict]:
        """Restore context from checkpoint"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get checkpoint info
        cursor.execute('''
            SELECT c.session_id, c.model_name, ct.context_data, ct.prompt, ct.response
            FROM session_checkpoints c
            JOIN context_tokens ct ON c.context_tokens_id = ct.id
            WHERE c.checkpoint_id = ?
        ''', (checkpoint_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        context_tokens = self.decompress_context(result[2])
        metadata = {
            'session_id': result[0],
            'model_name': result[1],
            'last_prompt': result[3],
            'last_response': result[4]
        }
        
        return context_tokens, metadata
    
    def query_with_context(self, model_name: str, prompt: str, 
                          context_tokens: Optional[List[int]] = None,
                          session_id: Optional[str] = None) -> Dict:
        """Query model with optional context tokens"""
        # Build request
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        
        # Add context if provided
        if context_tokens:
            request_data["context"] = context_tokens
        
        # Make request
        response = requests.post(
            f"{self.api_base}/api/generate",
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        result = response.json()
        
        # Save context if session provided
        if session_id and 'context' in result:
            self.save_context(
                session_id, model_name,
                result['context'], prompt, result['response']
            )
        
        return result
    
    def get_compression_stats(self) -> Dict:
        """Get statistics on context compression"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_contexts,
                SUM(context_size) as total_tokens,
                SUM(compressed_size) as total_compressed,
                AVG(context_size) as avg_context_size,
                AVG(compressed_size) as avg_compressed_size
            FROM context_tokens
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result[0] > 0:
            compression_ratio = 1 - (result[2] / (result[1] * 4))  # Assuming 4 bytes per token
            return {
                'total_contexts': result[0],
                'total_tokens': result[1],
                'total_compressed_bytes': result[2],
                'avg_context_size': result[3],
                'avg_compressed_size': result[4],
                'compression_ratio': f"{compression_ratio:.1%}"
            }
        
        return {}


def demo_context_persistence():
    """Demonstrate context token persistence"""
    print("CONTEXT TOKEN PERSISTENCE DEMO")
    print("=" * 50)
    
    # Initialize system
    ctp = ContextTokenPersistence()
    model = "phi3:mini"
    session_id = f"demo_{int(time.time())}"
    
    print(f"\nSession ID: {session_id}")
    print(f"Model: {model}\n")
    
    # Test 1: Initial query without context
    print("1. Initial query (no context):")
    result1 = ctp.query_with_context(
        model, 
        "Hello! My name is Bob and I'm learning about distributed systems.",
        session_id=session_id
    )
    print(f"Response: {result1['response'][:150]}...")
    print(f"Context tokens: {len(result1.get('context', []))}")
    
    time.sleep(2)
    
    # Test 2: Follow-up with saved context
    print("\n\n2. Follow-up query (using saved context):")
    context, metadata = ctp.load_context(session_id, model)
    result2 = ctp.query_with_context(
        model,
        "What's my name and what am I learning about?",
        context_tokens=context,
        session_id=session_id
    )
    print(f"Response: {result2['response'][:200]}...")
    
    # Test 3: Create checkpoint
    print("\n\n3. Creating checkpoint...")
    checkpoint_id = ctp.create_checkpoint(session_id, model, "learned_name")
    print(f"Checkpoint created: {checkpoint_id}")
    
    # Test 4: New conversation branch
    print("\n\n4. Starting new conversation branch...")
    result3 = ctp.query_with_context(
        model,
        "Actually, I changed my mind. I'm more interested in quantum computing.",
        context_tokens=context,
        session_id=session_id
    )
    print(f"Response: {result3['response'][:150]}...")
    
    # Test 5: Restore checkpoint
    print("\n\n5. Restoring from checkpoint...")
    restored_context, checkpoint_meta = ctp.restore_checkpoint(checkpoint_id)
    result4 = ctp.query_with_context(
        model,
        "What was I originally interested in learning?",
        context_tokens=restored_context
    )
    print(f"Response: {result4['response'][:200]}...")
    
    # Show compression stats
    print("\n\n6. Compression Statistics:")
    stats = ctp.get_compression_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n\nDemo complete! Context tokens provide portable model state.")


if __name__ == "__main__":
    demo_context_persistence()