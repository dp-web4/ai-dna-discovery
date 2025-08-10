# Memory Sensor

## Overview

Memory is a temporal sensor that parses the past to contextualize the present. Unlike traditional memory systems that just store and retrieve, this memory actively contributes to the reality field by providing weighted historical context.

## Structure

```
memory/
├── README.md                 # This file
├── experiences/              # Raw experience storage
│   ├── YYYY-MM-DD/          # Daily experiences
│   │   ├── HH-MM-SS.json    # Timestamped snapshots
│   │   └── summary.json     # Daily summary
│   └── index.json           # Experience index
├── patterns/                 # Recognized patterns
│   ├── spatial/             # Vision, motion patterns
│   ├── temporal/            # Time-based patterns
│   ├── contextual/          # Context-specific patterns
│   └── emergent/            # Newly discovered patterns
└── context/                  # Context snapshots
    ├── stable/              # Stable context memories
    ├── unstable/            # High-attention memories
    └── transitions/         # Context switch moments
```

## Memory as a Sensor

### Input
- Current sensor state
- Context markers
- Attention triggers

### Processing
1. **Pattern Matching**: Compare current state to stored patterns
2. **Relevance Filtering**: Weight memories by similarity to current context
3. **Trust Weighting**: Apply confidence based on memory accuracy history
4. **Temporal Decay**: Recent memories have higher weight

### Output
```python
{
    "relevant_memories": [
        {
            "timestamp": "2025-08-10T12:00:00",
            "pattern": "dual_camera_tracking",
            "confidence": 0.85,
            "context_similarity": 0.92,
            "data": {...}
        }
    ],
    "predicted_next": "likely_motion_left",
    "confidence": 0.75,
    "attention_suggestion": null
}
```

## Memory Types

### Episodic Memory
- Specific experiences with full context
- High detail, lower abstraction
- Used for novel situation comparison

### Semantic Memory
- Abstracted patterns and rules
- Low detail, high generalization
- Used for quick context recognition

### Working Memory
- Last 100 time steps
- Full resolution, no compression
- Immediate context comparison

## Trust and Relevance

### Trust Factors
- **Recency**: Newer memories initially more trusted
- **Reinforcement**: Repeatedly confirmed patterns gain trust
- **Context Match**: Memories from similar contexts weighted higher
- **Prediction Success**: Memories that led to correct predictions gain trust

### Relevance Calculation
```python
relevance = (
    context_similarity * 0.4 +
    temporal_proximity * 0.3 +
    pattern_strength * 0.2 +
    attention_match * 0.1
)
```

## Integration with Coherence Engine

The memory sensor provides:
1. **Historical Context**: What happened before in similar situations
2. **Pattern Recognition**: Known patterns in current sensor data
3. **Prediction**: Likely next states based on history
4. **Anomaly Detection**: When current state doesn't match any memory

## Storage Format

Each memory entry:
```json
{
    "timestamp": "ISO-8601",
    "context": {
        "state": "stable|moving|unstable",
        "attention_level": 0.0-1.0,
        "active_sensors": ["vision", "imu", ...]
    },
    "sensor_data": {
        "vision": {...},
        "imu": {...},
        "cognition": {...}
    },
    "patterns_detected": ["pattern_id_1", "pattern_id_2"],
    "prediction_made": "predicted_state",
    "prediction_accuracy": 0.0-1.0,
    "metadata": {
        "session_id": "uuid",
        "engine_version": "1.0.0"
    }
}
```

## Evolution

The memory system evolves by:
1. **Pattern Extraction**: Identifying repeated sequences
2. **Compression**: Abstracting details while preserving patterns
3. **Pruning**: Forgetting low-value memories
4. **Reinforcement**: Strengthening successful predictions

## Current Implementation

Starting with JSON-based storage for transparency and debugging. Future versions may use:
- SQLite for structured queries
- Vector database for similarity search
- Compression algorithms for long-term storage

The key is that memory isn't just storage - it's an active sensor contributing to the reality field.