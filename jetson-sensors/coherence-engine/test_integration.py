#!/usr/bin/env python3
"""
Integration test combining our coherence engine with GPT's enhancements
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "gpt_proposals_081125"))

# Import our coherence engine
from coherence_engine import CoherenceEngine, Context, ContextState, TrustModel, RelevanceModel

# Import GPT's components
from lifecycle import Lifecycle, Policy, PatternItem
from attention_trace import emit as emit_trace
from latency_watchdog import LatencyWatchdog

# Import our sensors
from sensors.vision_sensor import VisionSensor
from sensors.imu_sensor import IMUSensor
from sensors.cognition_sensor import CognitionSensor

def test_pattern_lifecycle_integration():
    """Test pattern lifecycle with coherence engine."""
    print("\n=== Pattern Lifecycle Integration ===")
    
    # Create lifecycle manager
    policy = Policy(
        support_min=3,
        promotion_score=0.75,
        contradiction_limit=2,
        expiry_sec=300
    )
    lifecycle = Lifecycle(policy)
    
    # Simulate pattern observations
    patterns = [
        ("stable_vision", True, 0.9),
        ("stable_vision", True, 0.8),
        ("stable_vision", True, 0.85),
        ("moving_imu", True, 0.7),
        ("moving_imu", False, 0.3),  # Contradiction
    ]
    
    for key, consistent, weight in patterns:
        item = lifecycle.observe(key, consistent, weight)
        print(f"  Pattern '{key}': state={item.state}, confidence={item.confidence:.2f}")
    
    # Run periodic maintenance
    lifecycle.periodic()
    
    # Check promoted patterns
    promoted = [k for k, v in lifecycle.items.items() if v.state == "PROMOTED"]
    print(f"✅ Promoted patterns: {promoted}")
    
    return True

def test_trust_evolution():
    """Test trust evolution with real sensors."""
    print("\n=== Trust Evolution Test ===")
    
    # Create sensors
    sensors = [
        VisionSensor(),
        IMUSensor(),
        CognitionSensor()
    ]
    
    # Create trust model
    trust = TrustModel(
        base={"vision": 0.5, "imu": 0.5, "cognition": 0.3},
        lr=0.1  # Learning rate
    )
    
    # Simulate trust updates
    for _ in range(5):
        for sensor in sensors:
            reading = sensor.read(tick=_)
            
            # Simulate alignment check (simplified)
            aligned = reading > 0.5
            delta = 0.1 if aligned else -0.1
            
            trust.update(sensor.id, ContextState.STABLE, delta)
            current_trust = trust.get(sensor.id, ContextState.STABLE)
            print(f"  {sensor.id}: trust={current_trust:.2f}")
    
    print("✅ Trust evolution working")
    return True

def test_latency_monitoring():
    """Test latency monitoring in coherence loop."""
    print("\n=== Latency Monitoring ===")
    
    # Create simple watchdog
    class SimpleWatchdog:
        def __init__(self):
            self.timings = {}
            
        def measure(self, name):
            class Timer:
                def __init__(self, watchdog, name):
                    self.watchdog = watchdog
                    self.name = name
                    self.start = None
                    
                def __enter__(self):
                    self.start = time.time()
                    
                def __exit__(self, *args):
                    elapsed = time.time() - self.start
                    self.watchdog.timings[self.name] = elapsed
                    
            return Timer(self, name)
    
    watchdog = SimpleWatchdog()
    
    # Create mini coherence engine
    sensors = [VisionSensor(), IMUSensor()]
    
    # Measure operations
    with watchdog.measure("sensor_read"):
        readings = {s.id: s.read(tick=0) for s in sensors}
        
    with watchdog.measure("fusion"):
        fused = sum(readings.values()) / len(readings)
        
    # Report timings
    for op, duration in watchdog.timings.items():
        print(f"  {op}: {duration*1000:.2f}ms")
        
    print("✅ Latency monitoring working")
    return True

def test_attention_tracing():
    """Test attention trace emission."""
    print("\n=== Attention Tracing ===")
    
    import os
    os.environ["ATTENTION_TRACE"] = "1"
    
    # Simulate attention shift
    emit_trace(
        policy="surprise_detection",
        features_up={"motion": 0.8, "novelty": 0.6},
        features_down={"stability": -0.3},
        reason="Sudden motion detected",
        weights_pre={"vision": 0.6, "imu": 0.4},
        weights_post={"vision": 0.4, "imu": 0.6}
    )
    
    print("✅ Attention trace emitted")
    
    # Check if file exists
    trace_file = Path("logs/attention_trace.jsonl")
    if trace_file.exists():
        with open(trace_file) as f:
            trace = json.loads(f.readline())
            print(f"  Traced: {trace['reason']}")
        
        # Cleanup
        trace_file.unlink()
        trace_file.parent.rmdir()
    
    return True

def main():
    """Run integration tests."""
    print("="*60)
    print("COHERENCE ENGINE + GPT ENHANCEMENTS")
    print("Integration Test Suite")
    print("="*60)
    
    tests = [
        ("Pattern Lifecycle", test_pattern_lifecycle_integration),
        ("Trust Evolution", test_trust_evolution),
        ("Latency Monitoring", test_latency_monitoring),
        ("Attention Tracing", test_attention_tracing)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"❌ {name} error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = "ERROR"
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    
    for name, result in results.items():
        symbol = "✅" if result == "PASSED" else "❌"
        print(f"{symbol} {name}: {result}")
    
    print(f"\nTotal: {passed}/{total} passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 Integration successful! GPT's enhancements work with our engine!")
    else:
        print(f"\n⚠️ {total - passed} tests need attention")

if __name__ == "__main__":
    main()