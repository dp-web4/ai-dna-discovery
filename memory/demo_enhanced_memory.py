#!/usr/bin/env python3
"""
Demo of the enhanced memory system
Shows confidence-aware memory storage, retrieval, and sensor integration
"""

import time
from datetime import datetime, timedelta
from enhanced_memory_system import HierarchicalMemory
from sensor_memory_integration import SensorMemoryIntegration, SensorReading

def print_section(title):
    """Pretty print section headers"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def demo_confidence_memory():
    """Demonstrate the enhanced memory system"""
    print("ENHANCED MEMORY SYSTEM DEMO v2.0")
    print("Featuring: Confidence scoring, hierarchical layers, and sensor integration")
    
    # Create memory system
    memory = HierarchicalMemory("demo_memory.db")
    session_id = f"demo_{int(time.time())}"
    
    # Part 1: Confidence-based storage
    print_section("1. CONFIDENCE-BASED MEMORY STORAGE")
    
    memories = [
        ("I am absolutely certain my name is Alice", 0.95, "identity"),
        ("I work as a software engineer at TechCorp", 0.85, "profession"),
        ("I think I might enjoy hiking on weekends", 0.6, "hobby"),
        ("Maybe I visited Paris last year... or was it Prague?", 0.3, "travel"),
        ("Vague memory of eating pizza yesterday", 0.4, "food")
    ]
    
    for content, confidence, concept in memories:
        result = memory.store_with_confidence(
            content=content,
            memory_type="semantic" if confidence > 0.7 else "episodic",
            session_id=session_id,
            source_confidence=confidence,
            metadata={'concept': concept}
        )
        
        if result:
            print(f"✓ Stored (conf {confidence:.2f}): {content[:40]}...")
        else:
            print(f"✗ Rejected (conf {confidence:.2f}): {content[:40]}...")
    
    # Part 2: Confidence-weighted retrieval
    print_section("2. CONFIDENCE-WEIGHTED RETRIEVAL")
    
    queries = ["name", "work", "Paris", "pizza"]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = memory.retrieve_with_confidence(query, limit=3)
        
        if results:
            for mem, weight in results:
                print(f"  → {mem.content[:50]}...")
                print(f"    Confidence: {mem.confidence.composite:.2f}, Weight: {weight:.2f}")
        else:
            print(f"  No results found")
    
    # Part 3: Sensor integration
    print_section("3. SENSOR-TO-MEMORY INTEGRATION")
    
    sensor_bridge = SensorMemoryIntegration(memory)
    
    # Simulate different sensor events
    sensor_events = [
        {
            'type': 'imu',
            'data': {
                'acceleration': [0.1, 0.2, 2.8],  # Sudden upward movement
                'orientation': {'roll': 5, 'pitch': 10, 'yaw': 180}
            },
            'confidence': 0.9,
            'description': "Sudden upward movement detected"
        },
        {
            'type': 'camera',
            'data': {
                'detected_objects': [
                    {'class': 'laptop', 'confidence': 0.94},
                    {'class': 'coffee_cup', 'confidence': 0.87}
                ],
                'scene_change': 0.3
            },
            'confidence': 0.85,
            'description': "Visual: laptop and coffee cup on desk"
        },
        {
            'type': 'audio',
            'data': {
                'transcription': 'Meeting starts in 5 minutes',
                'transcription_confidence': 0.82,
                'sound_events': []
            },
            'confidence': 0.8,
            'description': "Audio reminder about meeting"
        }
    ]
    
    for event in sensor_events:
        print(f"\n{event['description']}:")
        
        reading = SensorReading(
            sensor_type=event['type'],
            data=event['data'],
            confidence=event['confidence'],
            timestamp=datetime.now()
        )
        
        patterns = sensor_bridge.process_sensor_input({event['type']: reading})
        
        for pattern in patterns:
            print(f"  → {pattern.pattern_type}: {pattern.description}")
            print(f"    Stored as {pattern.sensor_type} memory (conf: {pattern.confidence:.2f})")
    
    # Part 4: Memory health and statistics
    print_section("4. MEMORY SYSTEM HEALTH CHECK")
    
    health = memory.get_memory_health()
    
    print(f"\nOverall Statistics:")
    print(f"  Total memories: {health['total_memories']}")
    print(f"  Average confidence: {health['average_confidence']:.2f}")
    print(f"  Working memory load: {health['working_memory_load']:.1%}")
    print(f"  Sensory buffer load: {health['sensory_buffer_load']:.1%}")
    
    print(f"\nMemory Type Distribution:")
    for mem_type, stats in health['memory_type_stats'].items():
        print(f"  {mem_type}: {stats['count']} memories (avg conf: {stats['average_confidence']:.2f})")
    
    if health['recommendations']:
        print(f"\nSystem Recommendations:")
        for rec in health['recommendations']:
            print(f"  ⚠ {rec}")
    
    # Part 5: Memory layers visualization
    print_section("5. HIERARCHICAL MEMORY LAYERS")
    
    print(f"\nCurrent Memory State:")
    print(f"  🧠 Consciousness Layer: Monitoring {health['total_memories']} memories")
    print(f"  📚 Semantic Memory: {len(memory.semantic_memory)} concepts")
    print(f"  🎬 Episodic Memory: {len(memory.episodic_memory)} experiences")
    print(f"  💭 Working Memory: {len(memory.working_memory)}/20 slots used")
    print(f"  👁️ Sensory Buffer: {len(memory.sensory_buffer)}/10 slots used")
    
    # Part 6: Time-based retrieval
    print_section("6. TEMPORAL AWARENESS")
    
    print("\nRecent memories (last 5 minutes):")
    recent = memory.retrieve_with_confidence(
        query="",  # Empty query to get all
        context={'time_window': timedelta(minutes=5)},
        limit=5
    )
    
    for mem, weight in recent[:3]:  # Show top 3
        age = datetime.now() - mem.timestamp
        print(f"  → {mem.content[:40]}... ({int(age.total_seconds())}s ago)")
    
    print("\n" + "="*60)
    print("Demo completed! Check 'demo_memory.db' for persistent storage.")
    print("="*60)

if __name__ == "__main__":
    demo_confidence_memory()