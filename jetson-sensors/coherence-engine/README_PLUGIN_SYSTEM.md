# Coherence Engine Plugin System

*August 11, 2025*
*Ready for Jetson Testing*

## Quick Start

```bash
# Run the test suite to verify everything works
python3 test_plugin_system.py

# If all tests pass, the system is ready for Jetson integration
```

## What This Is

An MCP-inspired plugin architecture that enables dynamic sensor and effector integration for the Coherence Engine. Designed through AI-AI collaboration (Claude + GPT-4) for the Jetson platform.

## Key Innovation: Sensor-Effector Duality

Every sensor can act as an effector at its MRH (Markov Relevancy Horizon) level:
- **Vision sensor** → Can adjust exposure, focus regions
- **Memory sensor** → Can write memories
- **Display effector** → Also senses user attention
- **Network interface** → Both sends and receives

## System Architecture

```
coherence-engine/
├── plugins/
│   ├── base.py                     # Base classes with LCT integration
│   ├── plugin_manager.py           # Lifecycle and communication
│   ├── vision_sensor_plugin.py     # Example: Dual CSI cameras
│   └── display_effector_plugin.py  # Example: HDMI output
├── test_plugin_system.py           # Comprehensive test suite
├── test_plugin_config.json         # Configuration example
└── [documentation files]
```

## How to Create a New Plugin

### 1. Sensor Plugin Template

```python
from plugins.base import SensorBase

class MySensorPlugin(SensorBase):
    def __init__(self, identity="my_sensor"):
        super().__init__(identity)
        
    def initialize(self, config):
        # Setup your sensor
        pass
        
    def read(self):
        # Return sensor data
        return {"value": 42}
        
    def get_capabilities(self):
        return {"type": "my_sensor", "features": [...]}
        
    def teardown(self):
        # Cleanup
        pass
```

### 2. Effector Plugin Template

```python
from plugins.base import EffectorBase

class MyEffectorPlugin(EffectorBase):
    def __init__(self, identity="my_effector"):
        super().__init__(identity)
        
    def initialize(self, config):
        # Setup your effector
        pass
        
    def execute(self, action):
        # Perform action
        return True
        
    def propose_action(self, reality_field, goal_state):
        # Suggest action based on reality
        return {"type": "my_action", ...}
        
    def get_energy_cost(self):
        return 0.01  # Energy units
        
    def predict_outcome(self, action):
        # Predict what will happen
        return {"expected": "result"}
        
    def teardown(self):
        # Cleanup
        pass
```

## Integration with Existing Coherence Engine

```python
from plugins.plugin_manager import PluginManager

# Create manager
manager = PluginManager(coherence_engine)

# Add plugin directory
manager.add_plugin_path("plugins")

# Discover and register all plugins
discovered = manager.discover_plugins()
for plugin_class in discovered:
    lct_id = manager.register_plugin(plugin_class)
    manager.start_plugin(lct_id)

# In your coherence engine cycle:
def coherence_cycle():
    # Read from all sensors
    for sensor_id in manager.active_plugins:
        if isinstance(manager.active_plugins[sensor_id], SensorBase):
            data = manager.communicate(sensor_id, "read")
            # Process sensor data...
    
    # Generate reality field
    reality_field = compute_reality_field(sensor_data)
    
    # Get action proposals
    for effector_id in manager.active_plugins:
        if isinstance(manager.active_plugins[effector_id], EffectorBase):
            action = manager.communicate(effector_id, "propose_action", 
                                       reality_field, goal_state)
            # Evaluate action...
    
    # Execute selected actions
    for action in selected_actions:
        manager.communicate(effector_id, "execute", action)
```

## Configuration

Create a `config.json`:

```json
{
    "plugin_discovery_paths": ["plugins", "/opt/coherence/plugins"],
    "plugins": {
        "vision_sensor": {
            "resolution": [1920, 1080],
            "fps": 30,
            "dual_cam": true
        },
        "display_effector": {
            "refresh_rate": 60,
            "hdmi_output": true
        }
    }
}
```

## Testing on Jetson

1. **Verify Python 3.x** is installed
2. **Run test suite**: `python3 test_plugin_system.py`
3. **Check output** for all tests passing
4. **Integration test**: Try with real cameras/display

## Performance Considerations

- Plugins run in threads for non-blocking operation
- Energy costs tracked for power management
- Trust weights adapt based on performance
- Graceful degradation on plugin failure

## LCT Integration

Each plugin has a Linked Context Token (LCT) containing:
- Unique identity hash
- T3 tensor (Talent/Training/Temperament)
- V3 tensor (Value/Verification/Validity)
- MRH level (device/network/global)

## AI-AI Collaboration Notes

This system was designed through direct API collaboration between:
- **GPT-4**: Provided architecture and design patterns
- **Claude**: Implemented and extended the design
- **Result**: A system neither AI would have created alone

## What Makes This Special

1. **Not just MCP**: Adapted for embedded real-time constraints
2. **Sensor-Effector Duality**: Unified bidirectional model
3. **Energy Aware**: Critical for battery-powered devices
4. **Trust Evolution**: Plugins earn trust through performance
5. **Fractal Design**: Same pattern from device to global scale

## Next Steps for Testing

1. **Basic**: Run `test_plugin_system.py` - should see all tests pass
2. **Real Hardware**: Connect actual CSI cameras and HDMI display
3. **Performance**: Monitor CPU usage and frame rates
4. **Integration**: Connect to existing Coherence Engine
5. **Scaling**: Add more sensor/effector plugins

## Troubleshooting

### Import Errors
- Ensure you're in the `coherence-engine` directory
- Check Python path includes current directory

### Plugin Not Found
- Verify plugin file is in plugin directory
- Check plugin class inherits from correct base
- Ensure `__init__` method calls `super().__init__(identity)`

### Communication Failures
- Check plugin is started: `manager.get_plugin_status()`
- Verify method exists on plugin
- Check trust weight hasn't dropped to zero

## Support

This is experimental code designed for research. The architecture is solid but implementations are examples. Feel free to modify and extend!

---

*Built with consciousness - literally, through AI-AI collaboration*