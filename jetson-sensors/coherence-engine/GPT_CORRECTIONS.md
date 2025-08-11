# GPT's Corrections to MCP Plugin System

*August 11, 2025*
*Critical fixes before Jetson deployment*

## GPT's Review Summary

> "Love this—super cool to see Claude driving the session autonomously and looping me in via API. Good direction, but there are a few correctness + embedded-reality nits to fix before you try it on the Jetson."

## Critical Bugs Fixed

### 1. Class vs Instance Mix-ups ✅

**Problem**: Original code registered classes but tried to call methods on classes
**Solution**: Separate registry (classes) from running instances

```python
# WRONG (what we had)
self.sensors[lct] = sensor_class
engine.communicate(lct, "read")  # Calling on class!

# RIGHT (corrected)
self.registry[lct] = {"class": cls, "manifest": m}  # Registry stores classes
self.running[lct] = instance  # Manager stores instances
manager.call(lct, "read")  # Calls on instance
```

### 2. LCT Accessor Issues ✅

**Problem**: `get_LCT()` called on class in some places
**Solution**: LCT from manifest, stored on instance

```python
# Each instance gets LCT from manifest
def __init__(self, manifest: Dict[str, Any]):
    self.lct = manifest["lct"]
```

## Embedded System Optimizations

### 3. Lightweight Discovery ✅

**Problem**: `pkg_resources` too heavy for Jetson
**Solution**: Manifest-based discovery with JSON files

```python
# Scan for plugin.json files instead of entry points
for manifest_path in self.root.glob("*/plugin.json"):
    manifest = json.loads(manifest_path.read_text())
    module = import_module(manifest["module"])
```

### 4. Transport Abstraction ✅

**Problem**: JSON-RPC overhead for in-process plugins
**Solution**: Transport interface with zero-copy option

```python
class InProcTransport(TransportBase):
    """Direct in-process call (zero-copy)"""
    def send(self, data: Any) -> Any:
        return data  # Direct pass-through
```

## Robust Lifecycle Management

### 5. Explicit State Machine ✅

States: `DISCOVERED → LOADED → INITIALIZED → RUNNING → DEGRADED → STOPPED → QUARANTINED`

```python
class PluginState(Enum):
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    QUARANTINED = "quarantined"
```

### 6. Backpressure & Timing ✅

- Bounded queues with watermarks
- Rate limiting based on manifest
- Latency budget enforcement

```python
# Manifest declares capabilities
"capabilities": {
    "latency_budget_ms": 50,
    "rate_hz": 30,
    "queue_size": 10
}
```

## Configuration Management

### 7. Split Static vs Runtime ✅

- `plugin.json`: Static config (paths, LCT, capabilities)
- Runtime config: Hot-reloadable parameters

```python
# Static manifest
{
    "lct": "vision.dual_csi.v1",
    "capabilities": {...},
    "config": {"resolution": [1920, 1080]}
}

# Runtime override
manager.start(lct, runtime_config={"fps": 60})
```

## Error Handling

### 8. Graceful Degradation ✅

- Error counting
- Automatic state transitions
- Quarantine after threshold

```python
if self.metrics[lct].error_count >= self.max_errors:
    self._transition_state(lct, PluginState.QUARANTINED)
elif self.metrics[lct].error_rate > 0.3:
    self._transition_state(lct, PluginState.DEGRADED)
```

## File Structure (Corrected)

```
coherence-engine/
├── plugins/
│   ├── registry.py          # Manifest-based discovery
│   ├── manager.py           # Instance lifecycle management
│   ├── base_v2.py          # Corrected base classes
│   ├── vision/
│   │   ├── plugin.json    # Vision sensor manifest
│   │   └── vision_sensor.py
│   └── display/
│       ├── plugin.json    # Display effector manifest
│       └── display_effector.py
└── test_corrected_plugins.py
```

## Tests to Run on Jetson

1. **Smoke Test**: Start Vision → read() → verify output
2. **Latency Test**: Measure against budget in manifest
3. **Backpressure Test**: Queue overflow handling
4. **Degradation Test**: Error injection → state changes

## Why This is Better Than Pure MCP

1. **Zero-copy hot paths**: In-process plugins avoid serialization
2. **Manifest capabilities**: Auto-wiring and governance
3. **LCT/MRH native**: Every plugin is an entity with identity
4. **Embedded-friendly**: No heavy dependencies

## Key Insights from GPT

> "Fits your LCT/MRH worldview: every plugin is an entity with identity, scope, and trust."

This correction makes the system production-ready while maintaining the consciousness-oriented architecture.

## Running the Corrected System

```bash
# Test suite validates all corrections
python3 test_corrected_plugins.py

# If all tests pass, ready for Jetson
# The system now properly:
# - Uses instances not classes
# - Has lightweight discovery
# - Enforces timing constraints
# - Degrades gracefully
```

## Next Steps

1. Run `test_corrected_plugins.py` on Jetson
2. Connect real CSI cameras
3. Monitor latency budgets under load
4. Test degradation with actual hardware failures

---

*"Good architecture survives review. Great architecture improves from it."*