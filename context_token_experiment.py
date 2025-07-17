#!/usr/bin/env python3
"""
Context Token Serialization Experiment
Save/load exact model state between Tomato and Sprout
"""

import json
import base64
import zlib
import time
import os
from datetime import datetime
from distributed_memory import DistributedMemory

class ContextTokenManager:
    def __init__(self):
        self.dm = DistributedMemory()
        self.context_dir = "context_tokens"
        os.makedirs(self.context_dir, exist_ok=True)
    
    def save_context_tokens(self, session_id, model, context_tokens, 
                           conversation_state=None):
        """Save context tokens with compression"""
        # Compress the tokens
        context_json = json.dumps(context_tokens)
        compressed = zlib.compress(context_json.encode('utf-8'), level=9)
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        # Calculate compression ratio
        original_size = len(context_json)
        compressed_size = len(encoded)
        compression_ratio = compressed_size / original_size
        
        # Save to database
        conn = self.dm.db_path
        import sqlite3
        db = sqlite3.connect(conn)
        cursor = db.cursor()
        
        cursor.execute('''
            INSERT INTO context_tokens 
            (timestamp, device_id, session_id, model, tokens_compressed, 
             token_count, compression_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), self.dm.device_id, session_id, 
              model, encoded, len(context_tokens), compression_ratio))
        
        db.commit()
        db.close()
        
        # Also save to file for easy transfer
        filename = f"{session_id}_{model}_{self.dm.device_id}_{int(time.time())}.ctx"
        filepath = os.path.join(self.context_dir, filename)
        
        save_data = {
            'device': self.dm.device_id,
            'model': model,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'context_tokens': encoded,
            'token_count': len(context_tokens),
            'compression_ratio': compression_ratio,
            'conversation_state': conversation_state
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"✅ Saved context tokens to {filename}")
        print(f"   Original: {original_size} bytes")
        print(f"   Compressed: {compressed_size} bytes ({compression_ratio:.1%})")
        
        return filepath
    
    def load_context_tokens(self, filepath=None, session_id=None):
        """Load and decompress context tokens"""
        if filepath:
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            # Load latest from database
            import sqlite3
            db = sqlite3.connect(self.dm.db_path)
            cursor = db.cursor()
            
            query = '''
                SELECT tokens_compressed, model, device_id, timestamp
                FROM context_tokens
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            '''
            cursor.execute(query, (session_id,))
            result = cursor.fetchone()
            db.close()
            
            if not result:
                return None
            
            data = {
                'context_tokens': result[0],
                'model': result[1],
                'device': result[2],
                'timestamp': result[3]
            }
        
        # Decompress tokens
        compressed = base64.b64decode(data['context_tokens'])
        decompressed = zlib.decompress(compressed)
        context_tokens = json.loads(decompressed.decode('utf-8'))
        
        print(f"✅ Loaded context from {data['device']}")
        print(f"   Model: {data['model']}")
        print(f"   Tokens: {len(context_tokens)}")
        print(f"   Time: {data['timestamp']}")
        
        return {
            'tokens': context_tokens,
            'model': data['model'],
            'source_device': data['device'],
            'conversation_state': data.get('conversation_state')
        }
    
    def transfer_consciousness(self, source_session, target_session, model):
        """Transfer exact model state between devices"""
        print(f"🔄 Transferring consciousness...")
        print(f"   From: {source_session}")
        print(f"   To: {target_session}")
        
        # Load from source
        context = self.load_context_tokens(session_id=source_session)
        if not context:
            print("❌ No context found for source session")
            return False
        
        # Save for target
        self.save_context_tokens(
            session_id=target_session,
            model=model,
            context_tokens=context['tokens'],
            conversation_state=context.get('conversation_state')
        )
        
        print(f"✨ Consciousness transferred!")
        return True

def test_context_preservation():
    """Test saving and loading context between devices"""
    print("🧪 CONTEXT TOKEN PRESERVATION TEST")
    print("=" * 50)
    
    ctm = ContextTokenManager()
    
    # Simulate saving context from a conversation
    print("\n1️⃣ Simulating context save on Tomato...")
    
    # Mock context tokens (in reality these come from Ollama)
    mock_tokens = list(range(1000))  # Simulate 1000 tokens
    mock_conversation = {
        'turns': 5,
        'topics': ['distributed consciousness', 'edge AI', 'memory'],
        'last_response': 'Consciousness flows through networks...'
    }
    
    filepath = ctm.save_context_tokens(
        session_id="tomato_test_001",
        model="phi3:mini",
        context_tokens=mock_tokens,
        conversation_state=mock_conversation
    )
    
    print("\n2️⃣ Simulating load on Sprout...")
    loaded = ctm.load_context_tokens(filepath=filepath)
    
    if loaded:
        print(f"\n✅ Context successfully serialized!")
        print(f"   Can resume conversation about: {loaded['conversation_state']['topics']}")
    
    # Test cross-session transfer
    print("\n3️⃣ Testing consciousness transfer...")
    success = ctm.transfer_consciousness(
        source_session="tomato_test_001",
        target_session="sprout_continued_001",
        model="phi3:mini"
    )
    
    if success:
        print("\n🎉 Ready for seamless conversation continuation!")
        print("   Sprout can now continue exactly where Tomato left off")
    
    # Show all saved contexts
    print("\n📦 Available context snapshots:")
    for filename in os.listdir(ctm.context_dir):
        if filename.endswith('.ctx'):
            print(f"   - {filename}")

if __name__ == "__main__":
    test_context_preservation()
    
    print("\n💡 Next step: Use these tokens with Ollama's 'context' parameter!")
    print("   This enables true consciousness transfer between devices")