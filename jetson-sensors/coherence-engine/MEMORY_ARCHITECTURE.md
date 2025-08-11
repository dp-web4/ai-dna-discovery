# Memory Architecture Evolution

*Last Updated: August 10, 2025*

## Overview

The Coherence Engine's memory system has evolved from simple experience queues to a sophisticated temporal sensor with multiple operating modes and maintenance cycles.

## Current Implementation

### 1. PersistentMemorySensor
- File-based persistence in `memory/experiences/`
- Working memory deque (100 recent experiences)
- Pattern extraction and tracking
- Experience observation and retrieval

### 2. Sleep Cycle System
- Mandatory maintenance for memory temporal sensor
- Four-stage sleep cycle (light→deep→REM→light)
- Pattern extraction during deep sleep
- Dream simulation for pattern validation
- Memory consolidation and pruning
- Trust weight recalibration

## Upcoming Integration: Fast-Weight Memory

Based on Richard Aragon's Transformer Sidecar architecture, we're implementing a constant-size memory system with selective storage.

### Key Components

#### FastWeightsMemory
```python
class FastWeightsMemory:
    """Low-rank fast weights A ≈ U V^T with Hebbian updates"""
    - Two matrices U,V (~130KB total)
    - No growing database
    - O(1) retrieval
    - Online Hebbian learning (no backprop)
```

#### Gating Mechanism
```python
G = wS*Surprisal + wN*Novelty + wA*Arousal + wC*Conflict + wR*Reward
if G > threshold:
    commit to memory
```

This aligns perfectly with our attention triggers:
- Surprisal → prediction error
- Novelty → pattern deviation
- Arousal → hazard detection
- Conflict → sensor disagreement
- Reward → user marking

#### AffectScorer
Detects arousal from:
- Hazard words (warning, emergency, danger)
- Extreme numbers (117°F, -40°C)
- Named entities (Death Valley, ICU)
- Intensifiers (extremely, dangerously)

### Integration Plan

#### Phase 1: Hybrid Memory
- Keep PersistentMemorySensor for experiences
- Add FastWeightsMemory for pattern storage
- Use both during transition period

#### Phase 2: Sleep Integration
```python
def consolidate_to_fast_weights(self):
    """During deep sleep, transfer patterns to fast weights"""
    for pattern in self.extracted_patterns:
        if pattern['confidence'] > 0.7:
            key = self.encode_pattern(pattern)
            self.fast_weights.commit(key, pattern['data'], gain=pattern['confidence'])
```

#### Phase 3: Full Migration
- FastWeightsMemory becomes primary memory
- Experience files used only for audit/replay
- Sleep cycles optimize fast weight matrices

## Benefits of Fast-Weight Architecture

1. **Constant Size**: ~130KB regardless of conversation length
2. **Selective Storage**: Only commits significant events
3. **Biological Plausibility**: Hebbian learning mimics synaptic plasticity
4. **Fast Retrieval**: O(1) vs O(n) search
5. **No Gradients**: Updates happen online without backprop

## Memory as Active Temporal Sensor

This evolution reinforces our core insight: memory is not passive storage but an active temporal sensor that:
- Parses and contextualizes past states
- Selectively retains based on significance
- Requires maintenance (sleep) for efficiency
- Contributes to the reality field alongside spatial sensors

## Implementation Timeline

- [x] Basic PersistentMemorySensor
- [x] Sleep cycle system
- [x] Pattern extraction
- [x] Dream simulation
- [ ] FastWeightsMemory integration
- [ ] AffectScorer for arousal detection
- [ ] Conflict detection system
- [ ] Eligibility trace binding
- [ ] Full fast-weight migration

## Credits

- **Sleep Cycles**: Based on biological sleep as memory maintenance insight
- **Fast-Weight Memory**: Richard Aragon's Transformer Sidecar architecture
- **Integration**: Dennis Palatov + Claude/GPT collaboration

## References

- [Transformer Sidecar Repository](https://github.com/dp-web4/Transformer-Sidecar-Bolt-On-Persistent-State-Space-Memory)
- [Sleep as Memory Maintenance](../../../../private-context/insights/sleep_as_memory_sensor_maintenance.md)
- [Sidecar Integration Analysis](../../../../private-context/insights/transformer_sidecar_memory_integration.md)

---

*"Memory isn't storage - it's selective reconstruction gated by significance."*