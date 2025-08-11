#!/usr/bin/env python3
"""
Test script for sleep cycle implementation
Tests the sleep cycle without requiring actual sensors
"""

import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque

# Mock classes for testing
@dataclass
class MockExperience:
    """Mock experience for testing"""
    timestamp: float
    context_state: str
    field_value: float
    trigger: str = None
    notes: dict = field(default_factory=dict)
    sensor_readings: dict = field(default_factory=dict)

class MockMemorySensor:
    """Mock memory sensor for testing sleep cycle"""
    def __init__(self):
        self.id = "memory"
        self.experiences = deque(maxlen=1000)
        self.working_memory = deque(maxlen=100)
        self.patterns = {
            'field_stability': 0.7,
            'trigger_rate': 0.2,
            'transition_STABLE->MOVING': 0.3
        }
        self.trust_weights = {'memory': 0.8, 'vision': 0.7}
        self.max_memory = 1000
        
        # Add some mock experiences
        for i in range(50):
            exp = MockExperience(
                timestamp=time.time() - (50 - i) * 60,
                context_state="STABLE" if i % 3 == 0 else "MOVING",
                field_value=0.5 + (i % 10) * 0.05,
                trigger="surprise" if i % 7 == 0 else None
            )
            self.experiences.append(exp)
            if i >= 40:  # Last 10 in working memory
                self.working_memory.append(exp)
    
    def _update_patterns(self):
        """Update patterns (mock)"""
        print("    - Updating memory patterns")
        self.patterns['field_stability'] = min(0.9, self.patterns['field_stability'] + 0.05)

class MockCoherenceEngine:
    """Mock coherence engine for testing"""
    def __init__(self):
        self.context = MockContext()
        self.sensors = {'memory': MockMemorySensor(), 'vision': None, 'imu': None}
        self.trust_model = MockTrustModel()
    
    def pause_external_sensors(self):
        print("    - Pausing external sensors")
    
    def resume_external_sensors(self, gradual=True):
        print(f"    - Resuming external sensors (gradual={gradual})")
    
    def simulate_scenario(self, elements):
        return {'coherence': 0.6 + len(elements) * 0.01}

class MockContext:
    """Mock context for testing"""
    def __init__(self):
        self.state = "STABLE"
        self.last_trigger = None

class MockTrustModel:
    """Mock trust model"""
    def __init__(self):
        self.base = {'memory': 0.8, 'vision': 0.7, 'imu': 0.75}

# Import the actual sleep cycle
from sleep_cycle import SleepCycle, SleepMetrics, DreamScenario

def test_sleep_metrics():
    """Test sleep metrics calculation"""
    print("\n=== Testing Sleep Metrics ===")
    
    memory = MockMemorySensor()
    engine = MockCoherenceEngine()
    
    # Create sleep cycle
    sleep_cycle = SleepCycle(memory, engine, sleep_dir="test_sleep")
    
    # Get metrics
    metrics = sleep_cycle.get_metrics()
    
    print(f"Retrieval latency: {metrics.retrieval_latency:.3f}")
    print(f"Pattern accuracy: {metrics.pattern_accuracy:.3f}")
    print(f"Memory pressure: {metrics.memory_pressure:.3f}")
    print(f"Trust staleness: {metrics.trust_staleness:.1f} hours")
    print(f"Experience backlog: {metrics.experience_backlog}")
    print(f"Time awake: {metrics.time_awake:.1f} hours")
    print(f"Sleep urgency: {metrics.sleep_urgency():.3f}")
    
    return metrics

def test_sleep_stages():
    """Test individual sleep stages"""
    print("\n=== Testing Sleep Stages ===")
    
    memory = MockMemorySensor()
    engine = MockCoherenceEngine()
    
    sleep_cycle = SleepCycle(memory, engine, sleep_dir="test_sleep")
    
    # Test light sleep 1
    print("\n1. Light Sleep (Stage 1):")
    sleep_cycle.light_sleep_1()
    
    # Test deep sleep
    print("\n2. Deep Sleep:")
    sleep_cycle.deep_sleep()
    
    # Test REM sleep
    print("\n3. REM Sleep:")
    sleep_cycle.rem_sleep()
    
    # Test light sleep 2
    print("\n4. Light Sleep (Stage 2):")
    sleep_cycle.light_sleep_2()
    
    print("\n✓ All sleep stages completed successfully")

def test_dream_generation():
    """Test dream scenario generation"""
    print("\n=== Testing Dream Generation ===")
    
    memory = MockMemorySensor()
    engine = MockCoherenceEngine()
    
    sleep_cycle = SleepCycle(memory, engine, sleep_dir="test_sleep")
    
    # Generate some dreams
    for i in range(3):
        dream = sleep_cycle.generate_dream_scenario()
        print(f"\nDream {i+1}:")
        print(f"  - Elements: {len(dream.elements)}")
        print(f"  - Mutations: {dream.mutations}")
        print(f"  - Emotional amplitude: {dream.emotional_amplitude:.2f}")
        print(f"  - Physics relaxed: {dream.physical_constraints_relaxed}")
        
        # Test the dream
        validation = sleep_cycle.test_dream_scenario(dream)
        print(f"  - Coherence: {validation['coherence']:.3f}")
        print(f"  - Patterns confirmed: {len(validation['patterns_confirmed'])}")
        print(f"  - Patterns violated: {len(validation['patterns_violated'])}")

def test_full_sleep_cycle():
    """Test a complete sleep cycle"""
    print("\n=== Testing Full Sleep Cycle ===")
    
    memory = MockMemorySensor()
    engine = MockCoherenceEngine()
    
    # Manually set last_sleep to simulate being awake for a while
    sleep_cycle = SleepCycle(memory, engine, sleep_dir="test_sleep")
    from datetime import timedelta
    sleep_cycle.last_sleep = datetime.now() - timedelta(hours=18)
    
    # Check if should sleep
    should_sleep = sleep_cycle.should_sleep()
    print(f"Should sleep: {should_sleep}")
    
    if should_sleep:
        print("\nEntering sleep cycle...")
        sleep_cycle.enter_sleep()
        print("\n✓ Sleep cycle completed successfully")
    
    # Check post-sleep metrics
    post_metrics = sleep_cycle.get_metrics()
    print(f"\nPost-sleep urgency: {post_metrics.sleep_urgency():.3f}")

def test_memory_consolidation():
    """Test memory consolidation functions"""
    print("\n=== Testing Memory Consolidation ===")
    
    memory = MockMemorySensor()
    engine = MockCoherenceEngine()
    sleep_cycle = SleepCycle(memory, engine, sleep_dir="test_sleep")
    
    # Add more similar experiences for consolidation
    for i in range(20):
        exp = MockExperience(
            timestamp=time.time() + i,
            context_state="STABLE",
            field_value=0.51,  # Very similar values
            trigger=None
        )
        memory.experiences.append(exp)
    
    initial_count = len(memory.experiences)
    print(f"Initial experience count: {initial_count}")
    
    # Test pattern extraction
    patterns = sleep_cycle.extract_patterns()
    print(f"Extracted {len(patterns)} patterns")
    for p in patterns:
        print(f"  - {p.get('type', 'unknown')}: {len(p.get('data', {}))} items")
    
    # Test consolidation
    consolidated = sleep_cycle.consolidate_similar()
    print(f"Consolidated {consolidated} similar memories")
    print(f"Final experience count: {len(memory.experiences)}")
    
    # Test pruning
    pruned = sleep_cycle.prune_irrelevant(threshold=0.2)
    print(f"Pruned {pruned} irrelevant memories")
    print(f"Final count after pruning: {len(memory.experiences)}")

def cleanup_test_files():
    """Clean up test files"""
    test_dir = Path("test_sleep")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print("\n✓ Cleaned up test files")

def main():
    """Run all tests"""
    print("=" * 60)
    print("SLEEP CYCLE TEST SUITE")
    print("=" * 60)
    
    try:
        # Run tests
        test_sleep_metrics()
        test_sleep_stages()
        test_dream_generation()
        test_memory_consolidation()
        test_full_sleep_cycle()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cleanup_test_files()

if __name__ == "__main__":
    main()