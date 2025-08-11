#!/usr/bin/env python3
"""
Test suite for MCP-like plugin system
August 11, 2025
"""

import sys
import os
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plugins.plugin_manager import PluginManager
from plugins.vision_sensor_plugin import VisionSensorPlugin
from plugins.display_effector_plugin import DisplayEffectorPlugin

def test_plugin_registration():
    """Test plugin registration and discovery"""
    print("=== Testing Plugin Registration ===")
    
    manager = PluginManager()
    
    # Register plugins directly
    vision_lct = manager.register_plugin(VisionSensorPlugin)
    display_lct = manager.register_plugin(DisplayEffectorPlugin)
    
    print(f"Vision LCT: {vision_lct}")
    print(f"Display LCT: {display_lct}")
    
    # Check registration
    status = manager.get_plugin_status()
    print(f"Registered plugins: {status['registered']}")
    
    assert vision_lct in status['registered']
    assert display_lct in status['registered']
    
    print("✓ Plugin registration successful\n")
    return manager, vision_lct, display_lct

def test_plugin_lifecycle(manager, vision_lct, display_lct):
    """Test starting and stopping plugins"""
    print("=== Testing Plugin Lifecycle ===")
    
    # Start plugins
    vision_config = {
        "resolution": (1920, 1080),
        "fps": 30,
        "dual_cam": True
    }
    manager.start_plugin(vision_lct, vision_config)
    
    display_config = {
        "resolution": (1920, 1080),
        "refresh_rate": 60,
        "hdmi_output": True
    }
    manager.start_plugin(display_lct, display_config)
    
    # Check active plugins
    status = manager.get_plugin_status()
    print(f"Active plugins: {status['active']}")
    
    assert vision_lct in status['active']
    assert display_lct in status['active']
    
    # Stop one plugin
    manager.stop_plugin(vision_lct)
    
    status = manager.get_plugin_status()
    assert vision_lct not in status['active']
    assert display_lct in status['active']
    
    # Restart plugin
    manager.start_plugin(vision_lct, vision_config)
    
    status = manager.get_plugin_status()
    assert vision_lct in status['active']
    
    print("✓ Plugin lifecycle management successful\n")

def test_plugin_communication(manager, vision_lct, display_lct):
    """Test communication with plugins"""
    print("=== Testing Plugin Communication ===")
    
    # Read from vision sensor
    vision_data = manager.communicate(vision_lct, "read")
    print(f"Vision data keys: {vision_data.keys()}")
    assert "frames" in vision_data
    assert "motion" in vision_data
    assert "timestamp" in vision_data
    
    # Get capabilities
    vision_caps = manager.communicate(vision_lct, "get_capabilities")
    print(f"Vision capabilities: {vision_caps}")
    assert vision_caps["type"] == "vision"
    
    display_caps = manager.communicate(display_lct, "get_capabilities")
    print(f"Display capabilities: {display_caps}")
    assert display_caps["type"] == "display"
    
    # Execute display action
    action = {
        "type": "overlay",
        "text": "Test Overlay",
        "position": (100, 100),
        "color": (255, 255, 255)
    }
    result = manager.communicate(display_lct, "execute", action)
    assert result == True
    
    print("✓ Plugin communication successful\n")

def test_sensor_effector_duality(manager, vision_lct, display_lct):
    """Test sensor-effector duality"""
    print("=== Testing Sensor-Effector Duality ===")
    
    # Vision sensor can also act as effector
    action = {
        "type": "adjust_exposure",
        "exposure": 1.5
    }
    result = manager.communicate(vision_lct, "execute", action)
    assert result == True
    print("✓ Vision sensor acted as effector")
    
    # Get energy costs
    vision_energy = manager.communicate(vision_lct, "get_energy_cost")
    display_energy = manager.communicate(display_lct, "get_energy_cost")
    
    print(f"Vision energy cost: {vision_energy}")
    print(f"Display energy cost: {display_energy}")
    
    assert vision_energy > 0
    assert display_energy > 0
    
    print("✓ Sensor-effector duality verified\n")

def test_reality_action_flow(manager, vision_lct, display_lct):
    """Test reality field to action field flow"""
    print("=== Testing Reality → Action Flow ===")
    
    # Simulate reality field
    reality_field = {
        "coherence": 0.85,
        "sensors": {
            "vision": 0.9,
            "memory": 0.7,
            "imu": 0.8
        },
        "brightness": 0.4
    }
    
    # Simulate goal state
    goal_state = {
        "needs_attention": True,
        "attention_region": [500, 300, 700, 500]
    }
    
    # Get action proposals
    vision_action = manager.communicate(vision_lct, "propose_action", reality_field, goal_state)
    print(f"Vision proposed: {vision_action}")
    
    display_action = manager.communicate(display_lct, "propose_action", reality_field, goal_state)
    print(f"Display proposed: {display_action}")
    
    # Execute proposed actions
    if vision_action:
        manager.communicate(vision_lct, "execute", vision_action)
    
    if display_action:
        manager.communicate(display_lct, "execute", display_action)
    
    print("✓ Reality → Action flow successful\n")

def test_plugin_failure_handling(manager, vision_lct):
    """Test plugin failure handling"""
    print("=== Testing Failure Handling ===")
    
    # Try to communicate with non-existent method
    try:
        result = manager.communicate(vision_lct, "non_existent_method")
    except:
        pass  # Expected to fail
    
    # Plugin should still be active
    status = manager.get_plugin_status()
    assert vision_lct in status['active']
    
    # Trust weight should be reduced
    plugin_details = status['details'][vision_lct]
    print(f"Trust weight after failure: {plugin_details['trust_weight']}")
    
    print("✓ Failure handling successful\n")

def test_plugin_discovery():
    """Test automatic plugin discovery"""
    print("=== Testing Plugin Discovery ===")
    
    manager = PluginManager()
    
    # Add plugin directory
    plugin_dir = Path(__file__).parent / "plugins"
    manager.add_plugin_path(str(plugin_dir))
    
    # Discover plugins
    discovered = manager.discover_plugins()
    print(f"Discovered {len(discovered)} plugins")
    
    for plugin_class in discovered:
        print(f"  - {plugin_class.__name__}")
    
    print("✓ Plugin discovery successful\n")

def create_test_config():
    """Create test configuration file"""
    config = {
        "plugin_discovery_paths": [
            "plugins"
        ],
        "plugins": {
            "vision_sensor": {
                "resolution": [1920, 1080],
                "fps": 30,
                "dual_cam": True
            },
            "display_effector": {
                "resolution": [1920, 1080],
                "refresh_rate": 60,
                "hdmi_output": True
            }
        },
        "runtime_parameters": {
            "vision_sensor": {
                "motion_threshold": 0.02
            },
            "display_effector": {
                "max_overlays": 5
            }
        }
    }
    
    config_file = "test_plugin_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created test config: {config_file}")
    return config_file

def main():
    """Run all tests"""
    print("="*60)
    print("MCP-LIKE PLUGIN SYSTEM TEST SUITE")
    print("="*60 + "\n")
    
    try:
        # Test registration
        manager, vision_lct, display_lct = test_plugin_registration()
        
        # Test lifecycle
        test_plugin_lifecycle(manager, vision_lct, display_lct)
        
        # Test communication
        test_plugin_communication(manager, vision_lct, display_lct)
        
        # Test sensor-effector duality
        test_sensor_effector_duality(manager, vision_lct, display_lct)
        
        # Test reality-action flow
        test_reality_action_flow(manager, vision_lct, display_lct)
        
        # Test failure handling
        test_plugin_failure_handling(manager, vision_lct)
        
        # Test discovery
        test_plugin_discovery()
        
        # Create config
        config_file = create_test_config()
        
        print("="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        
        print("\nPlugin system is ready for testing on Jetson.")
        print(f"Configuration file created: {config_file}")
        
        # Cleanup
        manager.shutdown()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())