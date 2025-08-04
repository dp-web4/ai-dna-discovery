# Enhanced Memory System v2.0

A confidence-aware, hierarchical memory system for AI agents with distributed synchronization capabilities.

## Overview

Building on the proven SQLite-based memory system that transforms stateless LLMs into stateful agents, this enhanced version introduces:

- **Confidence Scoring**: Web4-inspired quality metrics for every memory
- **Hierarchical Layers**: Sensory, working, episodic, semantic, and consciousness layers
- **Sensor Integration**: Bridge physical sensors to memory with confidence propagation
- **Distributed Sync**: Memory sharing across devices (Jetson, laptop, etc.)

## Architecture Components

### 1. Enhanced Memory System (`enhanced_memory_system.py`)
Core memory system with confidence awareness:
- Multi-dimensional confidence metrics (accuracy, relevance, reliability)
- Hierarchical memory layers with different retention policies
- Temporal decay for memory relevance
- SQLite backend with enhanced schema

### 2. Test Harness (`test_confidence_metrics.py`)
Comprehensive testing suite:
- Confidence scoring accuracy tests
- Retrieval precision measurements
- Temporal decay validation
- Memory consolidation testing
- Edge case handling

### 3. Integration Bridge (`memory_integration_bridge.py`)
Backwards compatibility with existing Phi3 memory:
- LLM confidence estimation
- Fact extraction with confidence
- Legacy database migration
- Context building with confidence weighting

### 4. Sensor Integration (`sensor_memory_integration.py`)
Bridges sensor confidence to memory:
- IMU motion pattern detection
- Camera object/scene recognition
- Audio transcription and events
- Bluetooth connection tracking

### 5. Distributed Sync (`distributed_memory_sync.py`)
Multi-device memory synchronization:
- Peer-to-peer memory sharing
- Conflict resolution using confidence
- Compressed delta transmission
- Automatic periodic synchronization

## Quick Start

### Basic Usage

```python
from enhanced_memory_system import HierarchicalMemory

# Create memory system
memory = HierarchicalMemory("my_memory.db")

# Store memory with confidence
memory.store_with_confidence(
    content="User's name is Alice",
    memory_type="semantic",
    session_id="session_123",
    source_confidence=0.9,
    metadata={'concept': 'identity'}
)

# Retrieve with confidence weighting
results = memory.retrieve_with_confidence(
    query="user name",
    context={'session_id': 'session_123'}
)

for memory_obj, weight in results:
    print(f"{memory_obj.content} (confidence: {memory_obj.confidence.composite:.2f})")
```

### Sensor Integration

```python
from sensor_memory_integration import SensorMemoryIntegration, SensorReading

# Create sensor bridge
sensor_bridge = SensorMemoryIntegration(memory)

# Process IMU data
imu_reading = SensorReading(
    sensor_type='imu',
    data={'acceleration': [0.1, 0.2, 2.5]},  # Sudden movement
    confidence=0.85,
    timestamp=datetime.now()
)

# Extract and store memorable patterns
patterns = sensor_bridge.process_sensor_input({'imu': imu_reading})
```

### Distributed Synchronization

```python
from distributed_memory_sync import DistributedMemory

# Setup distributed sync
dist_memory = DistributedMemory(
    device_id="laptop",
    memory_system=memory,
    sync_port=9999
)

# Add peer devices
dist_memory.add_peer("jetson", "192.168.1.100")

# Start automatic synchronization
dist_memory.start()
```

## Memory Confidence Model

Each memory has multi-dimensional confidence:

```python
@dataclass
class MemoryConfidence:
    accuracy: float      # Source reliability (0-1)
    relevance: float     # Current context relevance (0-1)
    reliability: float   # Historical reliability (0-1)
    composite: float     # Weighted average (0-1)
```

### Confidence Calculation

1. **Accuracy**: Based on source confidence (user input, sensor quality, LLM temperature)
2. **Relevance**: Computed against current goals and context
3. **Reliability**: Historical performance of similar memories
4. **Composite**: Weighted average (40% accuracy, 30% relevance, 30% reliability)

## Memory Layers

### Sensory Memory (< 1 minute)
- Raw sensor inputs
- Very short retention
- High volume, low processing

### Working Memory (< 20 items)
- Active context
- Current task state
- Quick access

### Episodic Memory (experiences)
- Conversations
- Events
- Temporal sequences

### Semantic Memory (facts)
- Extracted knowledge
- Concepts and relations
- Long-term retention

### Consciousness Layer (meta)
- Memory health monitoring
- Strategy adaptation
- Self-awareness

## Performance Metrics

- **Recall Accuracy**: >85% (improved from 67-100%)
- **Confidence Prediction**: >75% accuracy
- **Sync Latency**: <5 seconds
- **Memory Overhead**: <200MB per session
- **Compression Ratio**: <15% (improved from 21%)

## Testing

Run the comprehensive test suite:

```bash
python test_confidence_metrics.py
```

This will:
- Test confidence scoring accuracy
- Measure retrieval precision
- Validate temporal decay
- Test memory consolidation
- Generate detailed report

## Migration from Legacy System

To migrate existing Phi3 memory:

```python
from memory_integration_bridge import MemoryIntegrationBridge

bridge = MemoryIntegrationBridge()
bridge.migrate_from_legacy()
```

## Configuration

Key parameters in `enhanced_memory_system.py`:

```python
confidence_threshold = 0.5    # Minimum confidence to store
attention_budget = 1.0        # Resource allocation
decay_rate = 0.95            # Temporal decay factor
```

## Future Enhancements

### Core Improvements
- [ ] Embedding-based semantic search
- [ ] Advanced conflict resolution strategies
- [ ] Memory compression algorithms
- [ ] GPU-accelerated similarity computation
- [ ] Multi-modal memory fusion

### GPT-Suggested Enhancements

#### Immediate Value
1. **Temporal Memory Buffers**: Ring buffers for confidence time-series analysis
2. **Sensor Cross-Validation**: Detect contradictions between sensors to adjust confidence
3. **Memory as LCT Entity**: Memories have their own LCTs with T3/V3 profiles that guide:
   - Access frequency patterns
   - Validity/usefulness scoring
   - Compaction/discard decisions

#### Intelligence Layer
4. **Attention Dynamics**: Competitive inhibition between redundant sensors
5. **Predictive Confidence**: ML model to predict sensor failures from confidence history
6. **T3/V3 Sensor Profiles**: Each sensor has Talent/Training/Temperament characteristics

#### Experimental
7. **Saccadic Visual Sampling**: Biologically-inspired camera attention patterns
8. **Multi-Agent Consensus**: Distributed confidence weighting across AI agents

### Important Design Notes
- **LCT Clarification**: Sensors don't emit LCTs - they have LCTs that contextualize their output
- **Memory Entities**: Each memory could be an entity with its own T3/V3 profile affecting lifecycle
- **Lightweight Framework**: Need efficient implementation to avoid overhead

## Troubleshooting

### Low Confidence Memories
If memories are being rejected:
1. Check source confidence values
2. Adjust `confidence_threshold`
3. Review confidence calculation weights

### Sync Failures
For distributed sync issues:
1. Verify network connectivity
2. Check firewall settings for port 9999
3. Ensure peer device IDs are unique

### Performance Issues
If retrieval is slow:
1. Check database indices exist
2. Limit retrieval count
3. Use more specific queries