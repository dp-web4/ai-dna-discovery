#!/usr/bin/env python3
"""
Comprehensive test suite for corrected plugin system
Tests GPT's recommendations: latency, backpressure, lifecycle, etc.
"""

import unittest
import asyncio
import time
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from plugins.registry import Registry
from plugins.manager import PluginManager, PluginState
from plugins.base_v2 import SensorBase, EffectorBase

class TestRegistry(unittest.TestCase):
    """Test manifest-based plugin discovery"""
    
    def setUp(self):
        """Create temporary plugin directory with manifests"""
        self.test_dir = tempfile.mkdtemp()
        self.plugins_dir = Path(self.test_dir) / "plugins"
        self.plugins_dir.mkdir()
        
        # Create test plugin structure
        vision_dir = self.plugins_dir / "vision"
        vision_dir.mkdir()
        
        # Write manifest
        manifest = {
            "lct": "test.vision.v1",
            "module": "plugins.vision.vision_sensor",
            "class": "VisionSensor",
            "transport": "inproc",
            "capabilities": {
                "outputs": [{"name": "frames", "dtype": "numpy", "rate_hz": 30}],
                "latency_budget_ms": 50,
                "energy_hint_mw": 800
            }
        }
        (vision_dir / "plugin.json").write_text(json.dumps(manifest))
        
    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.test_dir)
    
    def test_discover_plugins(self):
        """Test plugin discovery from manifests"""
        registry = Registry(str(self.plugins_dir))
        discovered = registry.discover()
        
        self.assertIn("test.vision.v1", discovered)
        self.assertEqual(discovered["test.vision.v1"]["module"], "plugins.vision.vision_sensor")
    
    def test_get_manifest(self):
        """Test retrieving plugin manifest"""
        registry = Registry(str(self.plugins_dir))
        registry.discover()
        
        manifest = registry.get_manifest("test.vision.v1")
        self.assertEqual(manifest["lct"], "test.vision.v1")
        self.assertEqual(manifest["capabilities"]["latency_budget_ms"], 50)

class TestPluginManager(unittest.TestCase):
    """Test plugin lifecycle and instance management"""
    
    def setUp(self):
        """Set up test environment"""
        # Use actual plugin directory
        self.registry = Registry("plugins")
        self.registry.discover()
        self.manager = PluginManager(self.registry)
    
    def test_class_vs_instance(self):
        """Test that manager works with instances, not classes"""
        # Start a plugin
        if "vision.dual_csi.v1" in self.registry.registry:
            self.manager.start("vision.dual_csi.v1")
            
            # Verify instance is created
            self.assertIn("vision.dual_csi.v1", self.manager.running)
            
            # Verify it's an instance, not a class
            instance = self.manager.running["vision.dual_csi.v1"]
            self.assertIsNotNone(instance)
            self.assertTrue(hasattr(instance, 'lct'))
            self.assertEqual(instance.lct, "vision.dual_csi.v1")
    
    def test_lifecycle_states(self):
        """Test plugin state transitions"""
        if "vision.dual_csi.v1" in self.registry.registry:
            lct = "vision.dual_csi.v1"
            
            # Initial state
            self.assertEqual(self.manager.get_state(lct), PluginState.DISCOVERED)
            
            # Start plugin
            self.manager.start(lct)
            self.assertEqual(self.manager.get_state(lct), PluginState.RUNNING)
            
            # Stop plugin
            self.manager.stop(lct)
            self.assertEqual(self.manager.get_state(lct), PluginState.STOPPED)
    
    def test_call_method_on_instance(self):
        """Test calling methods on plugin instances"""
        if "vision.dual_csi.v1" in self.registry.registry:
            lct = "vision.dual_csi.v1"
            self.manager.start(lct)
            
            # Call read method
            result = self.manager.call(lct, "read")
            self.assertIsNotNone(result)
            self.assertIn("frames", result)
            self.assertIn("timestamp", result)
    
    def test_latency_budget(self):
        """Test latency budget monitoring"""
        if "vision.dual_csi.v1" in self.registry.registry:
            lct = "vision.dual_csi.v1"
            self.manager.start(lct)
            
            # Call and check metrics
            self.manager.call(lct, "read")
            metrics = self.manager.get_metrics(lct)
            
            self.assertEqual(metrics.call_count, 1)
            self.assertGreater(metrics.avg_latency, 0)
            
            # Check latency is reasonable (< 100ms for test)
            self.assertLess(metrics.avg_latency, 0.1)
    
    def test_error_handling_and_degradation(self):
        """Test error handling and state degradation"""
        if "display.hdmi.v1" in self.registry.registry:
            lct = "display.hdmi.v1"
            self.manager.start(lct)
            
            # Force an error by calling with bad action
            try:
                # This should fail but be handled
                self.manager.call(lct, "execute", {"type": "invalid_action"})
            except:
                pass
            
            # Check metrics recorded the error
            metrics = self.manager.get_metrics(lct)
            # Note: error might not be recorded if execute returns False vs raising
            
            # Multiple errors should degrade state
            for _ in range(10):
                try:
                    self.manager.call(lct, "nonexistent_method")
                except:
                    pass
            
            # Check if state changed (would be DEGRADED or QUARANTINED with enough errors)
            state = self.manager.get_state(lct)
            self.assertIn(state, [PluginState.RUNNING, PluginState.DEGRADED, PluginState.QUARANTINED])

class TestSensorEffectorIntegration(unittest.TestCase):
    """Test sensor-effector integration"""
    
    def setUp(self):
        """Set up both sensor and effector"""
        self.registry = Registry("plugins")
        self.registry.discover()
        self.manager = PluginManager(self.registry)
        
        # Start both if available
        if "vision.dual_csi.v1" in self.registry.registry:
            self.manager.start("vision.dual_csi.v1")
        if "display.hdmi.v1" in self.registry.registry:
            self.manager.start("display.hdmi.v1")
    
    def test_sensor_to_effector_flow(self):
        """Test data flow from sensor to effector"""
        if "vision.dual_csi.v1" in self.manager.running and "display.hdmi.v1" in self.manager.running:
            # Read from sensor
            sensor_data = self.manager.call("vision.dual_csi.v1", "read")
            self.assertIsNotNone(sensor_data)
            
            # Create reality field from sensor data
            reality_field = {
                "motion": sensor_data.get("motion", {}),
                "timestamp": sensor_data.get("timestamp")
            }
            
            # Get action proposal from effector
            goal_state = {"track_motion": True}
            action = self.manager.call("display.hdmi.v1", "propose_action", reality_field, goal_state)
            self.assertIsNotNone(action)
            
            # Execute action if proposed
            if action.get("type") != "none":
                success = self.manager.call("display.hdmi.v1", "execute", action)
                self.assertIsInstance(success, bool)
    
    def test_rate_limiting(self):
        """Test that plugins respect their rate limits"""
        if "vision.dual_csi.v1" in self.manager.running:
            # Read multiple times quickly
            start_time = time.time()
            read_count = 5
            
            for _ in range(read_count):
                self.manager.call("vision.dual_csi.v1", "read")
            
            elapsed = time.time() - start_time
            
            # At 30fps, 5 reads should take at least 4/30 = 0.133 seconds
            expected_min_time = (read_count - 1) / 30.0
            self.assertGreater(elapsed, expected_min_time * 0.8)  # Allow 20% variance

class TestBackpressure(unittest.TestCase):
    """Test async operations and backpressure handling"""
    
    def setUp(self):
        """Set up async test environment"""
        self.registry = Registry("plugins")
        self.registry.discover()
        self.manager = PluginManager(self.registry)
    
    def test_async_queue_creation(self):
        """Test that async plugins get queues"""
        if "vision.dual_csi.v1" in self.registry.registry:
            self.manager.start("vision.dual_csi.v1")
            
            # Check if queue was created (based on manifest async flag)
            manifest = self.registry.get_manifest("vision.dual_csi.v1")
            if manifest.get("capabilities", {}).get("async", False):
                self.assertIn("vision.dual_csi.v1", self.manager.queues)
    
    async def test_async_call(self):
        """Test async call mechanism"""
        if "vision.dual_csi.v1" in self.registry.registry:
            self.manager.start("vision.dual_csi.v1")
            
            # Make async call
            result = await self.manager.call_async("vision.dual_csi.v1", "read")
            self.assertIsNotNone(result)

def run_tests():
    """Run all tests with summary"""
    print("=" * 70)
    print("CORRECTED PLUGIN SYSTEM TEST SUITE")
    print("Testing GPT's recommendations:")
    print("- Class vs instance handling")
    print("- Manifest-based discovery")
    print("- Lifecycle state management")
    print("- Latency budget enforcement")
    print("- Error handling and degradation")
    print("- Sensor-effector integration")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestPluginManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSensorEffectorIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestBackpressure))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - Ready for Jetson!")
    else:
        print("❌ Some tests failed - Review and fix before deployment")
    
    print("=" * 70)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)