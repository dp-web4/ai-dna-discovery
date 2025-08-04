# AI-DNA-Discovery Memory Architecture v2.0

## Executive Summary

Building on the proven SQLite-based memory system that transforms stateless LLMs into stateful agents, this architecture introduces:
- Hierarchical memory layers with confidence scoring
- Distributed memory synchronization across devices
- Web4-inspired quality-aware data persistence
- Integration with sensor confidence frameworks

## Core Architecture

### 1. Memory Hierarchy

```
┌─────────────────────────────────────────┐
│         Consciousness Layer             │
│    (Meta-cognitive awareness)           │
├─────────────────────────────────────────┤
│         Semantic Memory                 │
│    (Facts, concepts, relations)         │
├─────────────────────────────────────────┤
│         Episodic Memory                 │
│    (Experiences, conversations)         │
├─────────────────────────────────────────┤
│         Working Memory                  │
│    (Active context window)              │
├─────────────────────────────────────────┤
│         Sensory Memory                  │
│    (Raw inputs with confidence)         │
└─────────────────────────────────────────┘
```

### 2. Enhanced Database Schema

```sql
-- Core conversation storage (existing)
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    importance_score REAL DEFAULT 0.5,
    confidence_score REAL DEFAULT 1.0,  -- NEW
    context_hash TEXT                   -- NEW
);

-- Enhanced facts with confidence
CREATE TABLE facts (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    fact_value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    frequency INTEGER DEFAULT 1,
    source_quality REAL DEFAULT 1.0,    -- NEW
    temporal_relevance REAL DEFAULT 1.0, -- NEW
    last_accessed DATETIME               -- NEW
);

-- Memory confidence tracking
CREATE TABLE memory_confidence (
    id INTEGER PRIMARY KEY,
    memory_type TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    factors JSON  -- Store contributing factors
);

-- Cross-device synchronization
CREATE TABLE sync_log (
    id INTEGER PRIMARY KEY,
    device_id TEXT NOT NULL,
    sync_timestamp DATETIME,
    memory_delta TEXT,  -- Compressed changes
    sync_status TEXT
);
```

### 3. Confidence-Aware Memory System

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class MemoryConfidence:
    """Web4-inspired confidence metrics for memory"""
    accuracy: float          # How accurate is this memory?
    relevance: float        # How relevant to current context?
    reliability: float      # Historical reliability score
    composite: float        # Overall confidence score

class HierarchicalMemory:
    """Enhanced memory system with confidence awareness"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.confidence_threshold = 0.5
        self.attention_budget = 1.0
        
    def store_with_confidence(self, 
                            content: str, 
                            memory_type: str,
                            source_confidence: float = 1.0):
        """Store memory with confidence metadata"""
        # Compute confidence based on source and context
        confidence = self._compute_memory_confidence(
            content, memory_type, source_confidence
        )
        
        # Store only if above threshold
        if confidence.composite > self.confidence_threshold:
            self._persist_memory(content, memory_type, confidence)
            
    def retrieve_with_confidence(self, 
                               query: str, 
                               context: Dict) -> List[tuple]:
        """Retrieve memories weighted by confidence"""
        # Get relevant memories
        candidates = self._search_memories(query)
        
        # Weight by confidence and relevance
        weighted_memories = []
        for memory, confidence in candidates:
            relevance = self._compute_relevance(memory, query, context)
            weight = confidence * relevance
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
        
        # Compute relevance to current goals
        relevance = self._assess_goal_relevance(content)
        
        # Check historical reliability of similar memories
        reliability = self._check_historical_reliability(content, memory_type)
        
        # Composite score using weighted average
        composite = (0.4 * accuracy + 0.3 * relevance + 0.3 * reliability)
        
        return MemoryConfidence(accuracy, relevance, reliability, composite)
```

### 4. Distributed Memory Synchronization

```python
class DistributedMemory:
    """Synchronize memory across multiple devices"""
    
    def __init__(self, device_id: str, sync_interval: int = 300):
        self.device_id = device_id
        self.sync_interval = sync_interval
        self.peers = []  # List of peer devices
        
    def sync_memories(self):
        """Synchronize memories with peer devices"""
        # Get local changes since last sync
        local_delta = self._get_memory_delta()
        
        # Exchange with peers
        for peer in self.peers:
            # Send our changes
            peer_delta = self._exchange_delta(peer, local_delta)
            
            # Merge peer changes with conflict resolution
            self._merge_memories(peer_delta)
            
    def _resolve_conflicts(self, local_memory, remote_memory):
        """Resolve conflicts using confidence scores"""
        # Higher confidence wins
        if local_memory.confidence > remote_memory.confidence:
            return local_memory
        elif remote_memory.confidence > local_memory.confidence:
            return remote_memory
        else:
            # Equal confidence - use timestamp
            return max(local_memory, remote_memory, 
                      key=lambda m: m.timestamp)
```

### 5. Integration with Sensor Confidence

```python
class SensorMemoryIntegration:
    """Bridge sensor confidence to memory confidence"""
    
    def process_sensor_input(self, sensor_data: Dict, sensor_confidence: Dict):
        """Store sensor data with appropriate confidence"""
        # Extract memorable patterns from sensor data
        patterns = self._extract_patterns(sensor_data)
        
        # Weight by sensor confidence
        for pattern in patterns:
            # Memory confidence inherits from sensor confidence
            memory_confidence = sensor_confidence.get(pattern.sensor_type, 0.5)
            
            # Store in appropriate memory layer
            if pattern.is_immediate:
                self.sensory_memory.store(pattern, memory_confidence)
            elif pattern.is_significant:
                self.episodic_memory.store(pattern, memory_confidence)
                
    def _extract_patterns(self, sensor_data: Dict) -> List:
        """Extract memorable patterns from raw sensor data"""
        patterns = []
        
        # Motion patterns from IMU
        if 'imu' in sensor_data:
            motion_pattern = self._analyze_motion(sensor_data['imu'])
            if motion_pattern.is_significant:
                patterns.append(motion_pattern)
                
        # Visual patterns from camera
        if 'camera' in sensor_data:
            visual_pattern = self._analyze_visual(sensor_data['camera'])
            if visual_pattern.is_memorable:
                patterns.append(visual_pattern)
                
        return patterns
```

### 6. Consciousness Layer

```python
class ConsciousnessLayer:
    """Meta-cognitive awareness of memory state"""
    
    def __init__(self, memory_system: HierarchicalMemory):
        self.memory_system = memory_system
        self.awareness_state = {}
        
    def assess_memory_health(self) -> Dict:
        """Evaluate overall memory system health"""
        health_metrics = {
            'working_memory_load': self._assess_working_memory(),
            'episodic_recall_rate': self._test_episodic_recall(),
            'semantic_coherence': self._check_semantic_coherence(),
            'confidence_distribution': self._analyze_confidence_distribution()
        }
        
        # Identify areas needing attention
        health_metrics['recommendations'] = self._generate_recommendations(health_metrics)
        
        return health_metrics
        
    def adapt_strategy(self, context: Dict):
        """Adapt memory strategy based on confidence"""
        # Get current memory confidence levels
        confidence_levels = self.memory_system.get_confidence_summary()
        
        if confidence_levels['average'] < 0.3:
            # Low confidence - switch to exploration mode
            self._enable_exploration_mode()
        elif confidence_levels['variance'] > 0.5:
            # High variance - focus on reliable sources
            self._focus_reliable_sources()
        else:
            # Normal operation
            self._standard_operation()
```

## Implementation Roadmap

### Phase 1: Enhanced Core (Week 1)
- Upgrade existing SQLite schema with confidence fields
- Implement confidence computation algorithms
- Add Web4-inspired quality metrics

### Phase 2: Hierarchical Layers (Week 2)
- Implement sensory memory buffer
- Create consciousness assessment layer
- Add temporal relevance decay

### Phase 3: Distribution (Week 3)
- Build device synchronization protocol
- Implement conflict resolution
- Create memory delta compression

### Phase 4: Integration (Week 4)
- Connect to sensor confidence framework
- Integrate with HRM model
- Deploy on Jetson and laptop

## Performance Targets

- Memory recall accuracy: >85% (up from 67-100%)
- Confidence prediction accuracy: >75%
- Sync latency: <5 seconds
- Memory overhead: <200MB per session
- Compression ratio: <15% (improved from 21%)

## Next Steps

1. Create initial implementation files
2. Set up test harness with confidence metrics
3. Implement core confidence algorithms
4. Begin integration with existing memory system