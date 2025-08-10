#!/usr/bin/env python3
"""
Test the Coherence Engine with Memory Sensor
"""

import sys
import time
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sensors'))

from coherence_engine import CoherenceEngine
from sensors.memory_sensor import MemorySensor

def main():
    """Test coherence engine with memory sensor"""
    
    print("="*60)
    print("COHERENCE ENGINE TEST")
    print("Testing Reality Field Generation with Memory Sensor")
    print("="*60)
    
    # Create engine with memory path
    memory_path = os.path.join(os.path.dirname(__file__), "memory")
    engine = CoherenceEngine(memory_path=memory_path)
    
    # Create and register memory sensor
    print("\nInitializing Memory Sensor...")
    memory_sensor = MemorySensor(memory_path=memory_path)
    engine.register_sensor(memory_sensor)
    
    print("\n" + "-"*60)
    print("Starting Coherence Engine Test")
    print("- Memory sensor will provide temporal context")
    print("- Engine will generate reality field from available sensors")
    print("- Context will shift based on patterns and triggers")
    print("-"*60)
    
    # Run for 30 seconds
    print("\nRunning for 30 seconds...")
    engine.run(duration=30)
    
    # Show summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    # Get sensor health
    for name, sensor in engine.sensors.items():
        health = sensor.get_health_status()
        print(f"\n{name}:")
        print(f"  Trust Score: {health['trust_score']:.2f}")
        print(f"  Success Rate: {health['success_rate']:.1%}")
        print(f"  Active: {health['is_active']}")
    
    # Show memory stats
    if 'memory_sensor' in engine.sensors:
        memory = engine.sensors['memory_sensor']
        context = memory.get_temporal_context()
        print(f"\nMemory Statistics:")
        print(f"  Working Memory: {context['working_memory_size']} experiences")
        print(f"  Known Patterns: {context['patterns_known']}")
    
    # Show context transitions
    transitions_path = os.path.join(memory_path, "context", "transitions")
    if os.path.exists(transitions_path):
        transitions = len(os.listdir(transitions_path))
        print(f"\nContext Transitions: {transitions}")
    
    # Clean shutdown
    engine.shutdown()
    
    print("\n" + "="*60)
    print("Test complete!")
    print(f"Experiences saved to: {memory_path}/experiences/")
    print(f"Patterns saved to: {memory_path}/patterns/")
    print("="*60)

if __name__ == "__main__":
    main()