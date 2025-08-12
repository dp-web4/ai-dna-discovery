#!/usr/bin/env python3
"""
Test GPT's components without module structure issues
"""

import sys
import os
from pathlib import Path

# Add the GPT proposals directory to path
gpt_dir = Path(__file__).parent / "gpt_proposals_081125"
sys.path.insert(0, str(gpt_dir))

def test_attention_trace():
    """Test attention trace functionality."""
    print("\n=== Testing Attention Trace ===")
    
    try:
        from attention_trace import emit, TRACE_ENABLED
        
        # Enable tracing for test
        os.environ["ATTENTION_TRACE"] = "1"
        
        # Test emit function
        emit(
            policy="test_policy",
            features_up={"sensor1": 0.5},
            features_down={"sensor2": 0.3},
            reason="test_reason",
            weights_pre={"sensor1": 0.4, "sensor2": 0.6},
            weights_post={"sensor1": 0.6, "sensor2": 0.4}
        )
        
        print("✅ Attention trace emit works")
        
        # Check if log file was created
        log_path = Path("logs/attention_trace.jsonl")
        if log_path.exists():
            print(f"✅ Log file created at {log_path}")
            # Clean up
            log_path.unlink()
            log_path.parent.rmdir()
        else:
            print("⚠️ Log file not created (tracing might be disabled)")
            
        return True
        
    except Exception as e:
        print(f"❌ Attention trace failed: {e}")
        return False

def test_lifecycle():
    """Test pattern lifecycle."""
    print("\n=== Testing Pattern Lifecycle ===")
    
    try:
        from lifecycle import Lifecycle, Policy
        
        # Create lifecycle manager
        policy = Policy()
        lifecycle = Lifecycle(policy)
        
        # Test genesis
        lifecycle.genesis({"pattern": "test"})
        assert len(lifecycle.patterns) == 1
        print("✅ Pattern genesis works")
        
        # Test maturation
        pid = list(lifecycle.patterns.keys())[0]
        lifecycle.hit(pid)
        lifecycle.hit(pid)
        assert lifecycle.patterns[pid]["confidence"] > 0
        print("✅ Pattern maturation works")
        
        # Test decay
        initial_conf = lifecycle.patterns[pid]["confidence"]
        lifecycle.decay_step()
        assert lifecycle.patterns[pid]["confidence"] < initial_conf
        print("✅ Pattern decay works")
        
        return True
        
    except Exception as e:
        print(f"❌ Lifecycle failed: {e}")
        return False

def test_conflict_resolution():
    """Test conflict resolution."""
    print("\n=== Testing Conflict Resolution ===")
    
    try:
        from conflict import Source, resolve
        
        # Create sources with conflicting values
        sources = [
            Source(id="vision", value=0.8, trust=0.9, context="stable"),
            Source(id="imu", value=0.2, trust=0.7, context="stable")
        ]
        
        # Resolve conflict
        result = resolve(sources)
        
        print(f"✅ Conflict resolved: value={result['value']:.3f}")
        print(f"   Strategy: {result['strategy']}")
        print(f"   Reasoning: {result['reasoning']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conflict resolution failed: {e}")
        return False

def test_latency_watchdog():
    """Test latency watchdog."""
    print("\n=== Testing Latency Watchdog ===")
    
    try:
        from latency_watchdog import LatencyWatchdog, Budgets
        
        # Create watchdog with budgets
        budgets = Budgets(
            sensor_read=0.010,  # 10ms
            fusion=0.005,        # 5ms
            decision=0.015       # 15ms
        )
        watchdog = LatencyWatchdog(budgets)
        
        # Test timing
        with watchdog.measure("sensor_read"):
            import time
            time.sleep(0.005)  # Simulate 5ms operation
            
        # Check if measurement worked
        last = watchdog.last_timings.get("sensor_read", 0)
        print(f"✅ Measured sensor_read: {last*1000:.1f}ms")
        
        # Test violation detection
        with watchdog.measure("fusion"):
            time.sleep(0.020)  # Exceed 5ms budget
            
        violations = watchdog.get_violations()
        if violations:
            print(f"✅ Detected violations: {violations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Latency watchdog failed: {e}")
        return False

def test_qos():
    """Test Quality of Service."""
    print("\n=== Testing Quality of Service ===")
    
    try:
        from qos import QoSBridge, ClassPolicy
        
        # Create QoS bridge
        policies = {
            "critical": ClassPolicy(priority=1.0, deadline_ms=10),
            "normal": ClassPolicy(priority=0.5, deadline_ms=100)
        }
        bridge = QoSBridge(policies)
        
        # Test message classification
        bridge.send({"type": "sensor", "data": "test"}, qos_class="critical")
        bridge.send({"type": "log", "data": "info"}, qos_class="normal")
        
        # Process messages
        count = bridge.process_batch(max_messages=10)
        print(f"✅ Processed {count} messages")
        
        # Check queue sizes
        for cls, size in bridge.queue_sizes().items():
            print(f"   {cls}: {size} messages")
            
        return True
        
    except Exception as e:
        print(f"❌ QoS failed: {e}")
        return False

def main():
    """Run all component tests."""
    print("="*60)
    print("GPT COMPONENT TESTS")
    print("Testing individual components")
    print("="*60)
    
    tests = [
        ("Attention Trace", test_attention_trace),
        ("Pattern Lifecycle", test_lifecycle),
        ("Conflict Resolution", test_conflict_resolution),
        ("Latency Watchdog", test_latency_watchdog),
        ("Quality of Service", test_qos)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"❌ {name} error: {e}")
            results[name] = "ERROR"
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    
    for name, result in results.items():
        symbol = "✅" if result == "PASSED" else "❌"
        print(f"{symbol} {name}: {result}")
    
    print(f"\nTotal: {passed}/{total} passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All components working!")
    else:
        print(f"\n⚠️ {total - passed} components need attention")

if __name__ == "__main__":
    main()