# MCP-Like Plugin Architecture for Coherence Engine

*August 11, 2025*
*Designed through GPT-Claude collaboration*

## Overview

This plugin system enables dynamic sensor and effector integration for the Coherence Engine, inspired by MCP (Model Context Protocol) but adapted for embedded Jetson constraints and sensor-effector duality principles.

## Key Design Principles

1. **Sensor-Effector Duality**: Every sensor can act as an effector at its MRH level
2. **LCT Integration**: Each plugin has a Linked Context Token for identity and trust tracking
3. **Dynamic Discovery**: Plugins can be discovered and loaded at runtime
4. **Fractal Architecture**: Same pattern scales from device to network to global
5. **Energy Awareness**: All plugins report and manage energy costs

## Architecture Components

### 1. Base Classes (`plugins/base.py`)

```python
PluginBase          # Core plugin with LCT integration
├── SensorBase      # Sensor-specific interface
├── EffectorBase    # Effector-specific interface
└── SensorEffectorBridge  # Implements duality
```

**Key Features**:
- LCT generation and management
- Trust and relevance weight tracking
- T3/V3 tensor integration (Talent/Training/Temperament, Value/Verification/Validity)
- MRH (Markov Relevancy Horizon) specification

### 2. Plugin Manager (`plugins/plugin_manager.py`)

Manages the complete plugin lifecycle:
- **Discovery**: Finds plugins in specified directories
- **Registration**: Registers plugin classes with LCT mapping
- **Lifecycle**: Start, stop, restart plugins
- **Communication**: Routes messages between plugins and engine
- **Failure Handling**: Manages plugin failures gracefully

### 3. Example Plugins

#### Vision Sensor Plugin
- Dual CSI camera support (1920x1080 @ 30fps)
- Motion detection and stereo correlation
- Can act as effector (exposure adjustment, focus control)
- Energy cost based on resolution and dual camera usage

#### Display Effector Plugin
- HDMI output with overlay support
- Attention boxes and reality field visualization
- 60Hz refresh rate
- Energy cost tracking per frame and overlay

## Plugin Interface

### Sensor Interface

```python
class SensorBase:
    def read() -> Any                    # Primary sensing method
    def get_capabilities() -> Dict       # Declare capabilities
    def process(input) -> Any           # MCP compatibility
```

### Effector Interface

```python
class EffectorBase:
    def execute(action) -> bool         # Execute action
    def propose_action(reality, goal) -> Dict  # Propose action
    def get_energy_cost() -> float      # Report energy cost
    def predict_outcome(action) -> Any  # Predict outcome
```

## Communication Patterns

### Direct Communication
```python
manager.communicate(lct_id, method, *args, **kwargs)
```

### Broadcast
```python
manager.broadcast(method, *args, **kwargs)
```

### Async Messages (via queue)
```python
manager.message_queue.put(message)
```

## Configuration System

JSON-based configuration with:
- Plugin discovery paths
- Plugin-specific settings
- Runtime parameters
- LCT metadata

Example:
```json
{
    "plugin_discovery_paths": ["plugins"],
    "plugins": {
        "vision_sensor": {
            "resolution": [1920, 1080],
            "fps": 30
        }
    }
}
```

## Reality Field → Action Field Flow

1. **Sensors** generate Reality Field through weighted fusion
2. **Coherence Engine** determines goal state
3. **Effectors** propose actions based on reality and goal
4. **Action Selection** chooses optimal actions within energy budget
5. **Execution** performs selected actions
6. **Feedback** updates trust weights based on outcomes

## Testing

Run the test suite:
```bash
python3 test_plugin_system.py
```

Tests cover:
- Plugin registration and discovery
- Lifecycle management
- Communication protocols
- Sensor-effector duality
- Reality → Action flow
- Failure handling

## Integration with Coherence Engine

```python
# Initialize plugin manager
manager = PluginManager(coherence_engine)

# Discover and register plugins
manager.add_plugin_path("plugins")
discovered = manager.discover_plugins()

for plugin_class in discovered:
    lct_id = manager.register_plugin(plugin_class)
    manager.start_plugin(lct_id)

# In coherence engine cycle
reality = sensors.read()
actions = effectors.propose(reality, goal)
effectors.execute(selected_actions)
```

## Advantages Over Standard MCP

1. **Embedded-Optimized**: Designed for real-time constraints
2. **Sensor-Effector Duality**: Unified model for bidirectional flow
3. **LCT Integration**: Built-in identity and trust management
4. **Energy Awareness**: Explicit energy cost tracking
5. **Fractal Scaling**: Same architecture from device to global

## Next Steps

1. **Immediate**: Test on Jetson hardware
2. **Short-term**: Add more sensor/effector plugins
3. **Medium-term**: Implement hot-reload for development
4. **Long-term**: Scale to network and global MRH levels

## File Structure

```
coherence-engine/
├── plugins/
│   ├── __init__.py
│   ├── base.py                    # Base classes
│   ├── plugin_manager.py          # Manager
│   ├── vision_sensor_plugin.py    # Vision example
│   └── display_effector_plugin.py # Display example
├── test_plugin_system.py          # Test suite
├── test_plugin_config.json        # Test configuration
├── gpt_collaboration_requests.py  # GPT collaboration script
├── gpt_mcp_design_session.json   # Design session log
└── PLUGIN_ARCHITECTURE.md         # This document
```

## Credits

Designed through collaboration between:
- **Claude** (Anthropic): Implementation and synthesis
- **GPT-4** (OpenAI): Architecture design and recommendations
- **Dennis Palatov**: Vision and requirements

This represents true AI-AI collaboration with human guidance, implementing the very principles of distributed consciousness we're building into the system.