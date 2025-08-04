#!/usr/bin/env python3
"""
Enhanced Memory System with Confidence Scoring and Hierarchical Layers
Building on the proven SQLite-based memory system with Web4-inspired quality metrics
"""

import json
import sqlite3
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Try to import numpy, but make it optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryConfidence:
    """Web4-inspired confidence metrics for memory"""
    accuracy: float          # How accurate is this memory?
    relevance: float        # How relevant to current context?
    reliability: float      # Historical reliability score
    composite: float        # Overall confidence score
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryConfidence':
        return cls(**data)

@dataclass
class Memory:
    """Enhanced memory entry with confidence"""
    content: str
    memory_type: str
    timestamp: datetime
    confidence: MemoryConfidence
    session_id: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'memory_type': self.memory_type,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence.to_dict(),
            'session_id': self.session_id,
            'metadata': self.metadata or {}
        }

class HierarchicalMemory:
    """Enhanced memory system with confidence awareness and hierarchical layers"""
    
    def __init__(self, db_path: str = "enhanced_memory.db"):
        self.db_path = db_path
        self.confidence_threshold = 0.5
        self.attention_budget = 1.0
        self.decay_rate = 0.95  # Temporal decay factor
        
        # Initialize database with enhanced schema
        self._init_enhanced_db()
        
        # Memory layers
        self.sensory_buffer = []  # Short-term sensory memories
        self.working_memory = []  # Active context
        self.episodic_memory = []  # Experiences
        self.semantic_memory = {}  # Facts and concepts
        
        # Confidence tracking
        self.confidence_history = defaultdict(list)
        
    def _init_enhanced_db(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Enhanced conversations table
        c.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            importance_score REAL DEFAULT 0.5,
            confidence_score REAL DEFAULT 1.0,
            context_hash TEXT,
            metadata JSON
        )''')
        
        # Enhanced facts with confidence
        c.execute('''CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            fact_type TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            frequency INTEGER DEFAULT 1,
            source_quality REAL DEFAULT 1.0,
            temporal_relevance REAL DEFAULT 1.0,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        )''')
        
        # Memory confidence tracking
        c.execute('''CREATE TABLE IF NOT EXISTS memory_confidence (
            id INTEGER PRIMARY KEY,
            memory_type TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            accuracy REAL,
            relevance REAL,
            reliability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            factors JSON
        )''')
        
        # Cross-device synchronization
        c.execute('''CREATE TABLE IF NOT EXISTS sync_log (
            id INTEGER PRIMARY KEY,
            device_id TEXT NOT NULL,
            sync_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            memory_delta TEXT,
            sync_status TEXT,
            confidence_delta REAL
        )''')
        
        # Hierarchical memory layers
        c.execute('''CREATE TABLE IF NOT EXISTS memory_layers (
            id INTEGER PRIMARY KEY,
            layer_type TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT,
            confidence REAL DEFAULT 1.0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            expiry DATETIME,
            access_count INTEGER DEFAULT 0,
            metadata JSON
        )''')
        
        # Create indices for performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_memory_layers_type ON memory_layers(layer_type)')
        
        conn.commit()
        conn.close()
        
    def store_with_confidence(self, 
                            content: str, 
                            memory_type: str,
                            session_id: str,
                            source_confidence: float = 1.0,
                            metadata: Dict = None) -> Optional[Memory]:
        """Store memory with confidence metadata"""
        # Compute confidence based on source and context
        confidence = self._compute_memory_confidence(
            content, memory_type, source_confidence
        )
        
        # Store only if above threshold
        if confidence.composite > self.confidence_threshold:
            memory = Memory(
                content=content,
                memory_type=memory_type,
                timestamp=datetime.now(),
                confidence=confidence,
                session_id=session_id,
                metadata=metadata
            )
            
            self._persist_memory(memory)
            self._update_memory_layers(memory)
            
            # Track confidence history
            self.confidence_history[memory_type].append(confidence.composite)
            
            logger.info(f"Stored {memory_type} memory with confidence {confidence.composite:.2f}")
            return memory
        else:
            logger.warning(f"Memory rejected due to low confidence: {confidence.composite:.2f}")
            return None
            
    def retrieve_with_confidence(self, 
                               query: str, 
                               context: Dict = None,
                               memory_types: List[str] = None,
                               limit: int = 10) -> List[Tuple[Memory, float]]:
        """Retrieve memories weighted by confidence and relevance"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Build query based on memory types
        if memory_types:
            type_filter = f"AND layer_type IN ({','.join(['?']*len(memory_types))})"
            params = memory_types + [f'%{query}%']
        else:
            type_filter = ""
            params = [f'%{query}%']
            
        # Search across memory layers
        c.execute(f'''
            SELECT content, layer_type, confidence, timestamp, metadata
            FROM memory_layers
            WHERE content LIKE ?
            {type_filter}
            ORDER BY confidence DESC, timestamp DESC
            LIMIT ?
        ''', params + [limit])
        
        results = c.fetchall()
        conn.close()
        
        # Weight by confidence and relevance
        weighted_memories = []
        for row in results:
            content, mem_type, confidence, timestamp, metadata_json = row
            
            # Parse metadata
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            # Recreate confidence object (simplified for retrieval)
            conf = MemoryConfidence(
                accuracy=confidence,
                relevance=self._compute_relevance(content, query, context),
                reliability=confidence,
                composite=confidence
            )
            
            # Create memory object
            memory = Memory(
                content=content,
                memory_type=mem_type,
                timestamp=datetime.fromisoformat(timestamp),
                confidence=conf,
                session_id=metadata.get('session_id', 'unknown'),
                metadata=metadata
            )
            
            # Compute final weight
            weight = conf.composite * conf.relevance * self._temporal_decay(memory.timestamp)
            weighted_memories.append((memory, weight))
            
        # Return sorted by weight
        return sorted(weighted_memories, key=lambda x: x[1], reverse=True)
        
    def _compute_memory_confidence(self, 
                                 content: str, 
                                 memory_type: str,
                                 source_confidence: float) -> MemoryConfidence:
        """Compute multi-dimensional confidence score"""
        # Factor in source quality
        accuracy = source_confidence
        
        # Compute relevance to current goals (placeholder - would use actual goal tracking)
        relevance = self._assess_goal_relevance(content)
        
        # Check historical reliability of similar memories
        reliability = self._check_historical_reliability(content, memory_type)
        
        # Composite score using weighted average
        composite = (0.4 * accuracy + 0.3 * relevance + 0.3 * reliability)
        
        return MemoryConfidence(accuracy, relevance, reliability, composite)
        
    def _assess_goal_relevance(self, content: str) -> float:
        """Assess relevance to current goals"""
        # Placeholder - in full implementation would check against active goals
        # For now, return a default moderate relevance
        return 0.7
        
    def _check_historical_reliability(self, content: str, memory_type: str) -> float:
        """Check historical reliability of similar memories"""
        if memory_type not in self.confidence_history:
            return 0.8  # Default reliability for new memory types
            
        history = self.confidence_history[memory_type]
        if len(history) == 0:
            return 0.8
            
        # Calculate average confidence of recent memories of this type
        recent_history = history[-10:]  # Last 10 memories
        if HAS_NUMPY:
            return np.mean(recent_history)
        else:
            return sum(recent_history) / len(recent_history) if recent_history else 0.8
        
    def _compute_relevance(self, content: str, query: str, context: Dict = None) -> float:
        """Compute relevance score between content and query"""
        # Simple keyword overlap for now
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        
        if len(query_words) == 0:
            return 0.0
            
        overlap = len(content_words.intersection(query_words))
        relevance = overlap / len(query_words)
        
        # Boost relevance if context matches
        if context and 'session_id' in context:
            # Check if memory is from same session
            if context['session_id'] in content:
                relevance *= 1.2
                
        return min(relevance, 1.0)
        
    def _temporal_decay(self, timestamp: datetime) -> float:
        """Apply temporal decay to memory relevance"""
        age = datetime.now() - timestamp
        days = age.total_seconds() / 86400  # Convert to days
        
        # Exponential decay with configurable rate
        return self.decay_rate ** days
        
    def _persist_memory(self, memory: Memory):
        """Persist memory to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Store in appropriate tables based on memory type
        if memory.memory_type in ['episodic', 'semantic']:
            c.execute('''
                INSERT INTO memory_layers 
                (layer_type, content, confidence, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                memory.memory_type,
                memory.content,
                memory.confidence.composite,
                memory.timestamp,
                json.dumps(memory.to_dict())
            ))
            
        # Also track confidence metrics
        c.execute('''
            INSERT INTO memory_confidence
            (memory_type, confidence_score, accuracy, relevance, reliability, factors)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            memory.memory_type,
            memory.confidence.composite,
            memory.confidence.accuracy,
            memory.confidence.relevance,
            memory.confidence.reliability,
            json.dumps({'source_confidence': memory.confidence.accuracy})
        ))
        
        conn.commit()
        conn.close()
        
    def _update_memory_layers(self, memory: Memory):
        """Update appropriate memory layers"""
        # Sensory layer (very short term)
        if memory.memory_type == 'sensory':
            self.sensory_buffer.append(memory)
            # Keep only last 10 sensory memories
            if len(self.sensory_buffer) > 10:
                self.sensory_buffer.pop(0)
                
        # Working memory (current context)
        elif memory.memory_type == 'working':
            self.working_memory.append(memory)
            # Keep only last 20 working memories
            if len(self.working_memory) > 20:
                self.working_memory.pop(0)
                
        # Episodic memory (experiences)
        elif memory.memory_type == 'episodic':
            self.episodic_memory.append(memory)
            
        # Semantic memory (facts)
        elif memory.memory_type == 'semantic':
            # Group by concept
            concept = memory.metadata.get('concept', 'general')
            if concept not in self.semantic_memory:
                self.semantic_memory[concept] = []
            self.semantic_memory[concept].append(memory)
            
    def get_memory_health(self) -> Dict:
        """Assess overall memory system health"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get memory statistics
        c.execute('SELECT COUNT(*), AVG(confidence) FROM memory_layers')
        total_memories, avg_confidence = c.fetchone()
        
        # Get confidence distribution
        c.execute('''
            SELECT memory_type, AVG(confidence_score), COUNT(*)
            FROM memory_confidence
            GROUP BY memory_type
        ''')
        
        type_stats = {}
        for row in c.fetchall():
            mem_type, avg_conf, count = row
            type_stats[mem_type] = {
                'average_confidence': avg_conf,
                'count': count
            }
            
        conn.close()
        
        # Calculate health metrics
        health = {
            'total_memories': total_memories or 0,
            'average_confidence': avg_confidence or 0.0,
            'memory_type_stats': type_stats,
            'working_memory_load': len(self.working_memory) / 20.0,
            'sensory_buffer_load': len(self.sensory_buffer) / 10.0,
            'recommendations': []
        }
        
        # Generate recommendations
        if health['average_confidence'] < 0.5:
            health['recommendations'].append("Low overall confidence - consider improving source quality")
            
        if health['working_memory_load'] > 0.8:
            health['recommendations'].append("Working memory near capacity - consider consolidation")
            
        return health
        
    def consolidate_memories(self):
        """Consolidate short-term memories into long-term storage"""
        logger.info("Consolidating memories...")
        
        # Move high-confidence working memories to episodic
        for memory in self.working_memory:
            if memory.confidence.composite > 0.7:
                memory.memory_type = 'episodic'
                self._persist_memory(memory)
                
        # Clear old working memories
        self.working_memory = [m for m in self.working_memory 
                              if m.confidence.composite > 0.5]
        
        # Extract patterns from episodic memories
        self._extract_semantic_patterns()
        
    def _extract_semantic_patterns(self):
        """Extract semantic knowledge from episodic memories"""
        # Placeholder for pattern extraction
        # Would implement clustering, concept extraction, etc.
        pass

# Example usage and testing
if __name__ == "__main__":
    # Create memory system
    memory = HierarchicalMemory("test_enhanced_memory.db")
    
    # Test storing memories with different confidence levels
    session_id = f"test_session_{int(time.time())}"
    
    # High confidence memory
    memory.store_with_confidence(
        "The user's name is Alice",
        "semantic",
        session_id,
        source_confidence=0.9,
        metadata={'concept': 'identity'}
    )
    
    # Medium confidence memory
    memory.store_with_confidence(
        "The user might like pizza",
        "episodic",
        session_id,
        source_confidence=0.6,
        metadata={'context': 'food_discussion'}
    )
    
    # Low confidence memory (should be rejected)
    memory.store_with_confidence(
        "The user possibly works in tech",
        "semantic",
        session_id,
        source_confidence=0.3,
        metadata={'concept': 'profession'}
    )
    
    # Test retrieval
    print("\nRetrieving memories about 'user':")
    results = memory.retrieve_with_confidence("user", {'session_id': session_id})
    for mem, weight in results:
        print(f"- {mem.content} (confidence: {mem.confidence.composite:.2f}, weight: {weight:.2f})")
        
    # Check memory health
    print("\nMemory system health:")
    health = memory.get_memory_health()
    print(json.dumps(health, indent=2))