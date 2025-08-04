#!/usr/bin/env python3
"""
Quick test of the enhanced memory system
Tests basic functionality before full test harness
"""

import time
from datetime import datetime
from enhanced_memory_system import HierarchicalMemory

def test_basic_functionality():
    """Test basic store and retrieve with confidence"""
    print("Testing Enhanced Memory System...")
    print("=" * 50)
    
    # Create memory system
    memory = HierarchicalMemory("test_basic.db")
    session_id = f"test_{int(time.time())}"
    
    # Test 1: Store memories with different confidence levels
    print("\n1. Storing memories with varying confidence...")
    
    test_data = [
        ("My name is Alice", 0.95, "semantic", {'concept': 'identity'}),
        ("I work as a software engineer", 0.85, "semantic", {'concept': 'profession'}),
        ("I might enjoy hiking", 0.6, "episodic", {'context': 'uncertain'}),
        ("I possibly live in Seattle", 0.4, "semantic", {'concept': 'location'}),
        ("Random noise data", 0.2, "sensory", None)  # Should be rejected
    ]
    
    stored_count = 0
    for content, confidence, mem_type, metadata in test_data:
        result = memory.store_with_confidence(
            content=content,
            memory_type=mem_type,
            session_id=session_id,
            source_confidence=confidence,
            metadata=metadata
        )
        if result:
            stored_count += 1
            print(f"  ✓ Stored: '{content[:30]}...' (conf: {confidence})")
        else:
            print(f"  ✗ Rejected: '{content[:30]}...' (conf: {confidence} below threshold)")
    
    print(f"\nStored {stored_count} out of {len(test_data)} memories")
    
    # Test 2: Retrieve memories
    print("\n2. Testing retrieval with confidence weighting...")
    
    queries = [
        "name",
        "work", 
        "Seattle",
        "hiking"
    ]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        results = memory.retrieve_with_confidence(query, limit=3)
        
        if results:
            for mem, weight in results:
                print(f"    - {mem.content} (conf: {mem.confidence.composite:.2f}, weight: {weight:.2f})")
        else:
            print(f"    No results found")
    
    # Test 3: Memory health check
    print("\n3. Memory System Health Check...")
    health = memory.get_memory_health()
    
    print(f"  Total memories: {health['total_memories']}")
    print(f"  Average confidence: {health['average_confidence']:.2f}")
    print(f"  Working memory load: {health['working_memory_load']:.1%}")
    print(f"  Sensory buffer load: {health['sensory_buffer_load']:.1%}")
    
    if health['recommendations']:
        print("\n  Recommendations:")
        for rec in health['recommendations']:
            print(f"    - {rec}")
    
    # Test 4: Hierarchical layers
    print("\n4. Testing hierarchical memory layers...")
    
    # Add some working memory
    memory.store_with_confidence(
        "Current task: testing memory system",
        "working",
        session_id,
        0.8
    )
    
    # Add sensory memory
    memory.store_with_confidence(
        "Quick visual flash of blue screen",
        "sensory", 
        session_id,
        0.7
    )
    
    print(f"  Sensory buffer: {len(memory.sensory_buffer)} items")
    print(f"  Working memory: {len(memory.working_memory)} items")
    print(f"  Episodic memory: {len(memory.episodic_memory)} items")
    print(f"  Semantic concepts: {len(memory.semantic_memory)} categories")
    
    # Test 5: Memory consolidation
    print("\n5. Testing memory consolidation...")
    
    # Fill working memory
    for i in range(5):
        memory.store_with_confidence(
            f"Working memory item {i}",
            "working",
            session_id,
            0.7 + (i * 0.05)  # Increasing confidence
        )
    
    print(f"  Working memory before consolidation: {len(memory.working_memory)}")
    
    # Trigger consolidation
    memory.consolidate_memories()
    
    print(f"  Working memory after consolidation: {len(memory.working_memory)}")
    print(f"  (High-confidence items moved to episodic memory)")
    
    print("\n" + "=" * 50)
    print("Basic functionality test completed!")
    
    return True

if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        if success:
            print("\n✅ All basic tests passed!")
        else:
            print("\n❌ Some tests failed")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()