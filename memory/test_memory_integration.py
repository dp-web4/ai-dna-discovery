#!/usr/bin/env python3
"""
Integration test for memory system with ollama
Tests the bridge between enhanced memory and LLM
"""

import time
import subprocess
from datetime import datetime
from memory_integration_bridge import MemoryIntegrationBridge

def check_ollama():
    """Check if ollama is available"""
    try:
        result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama found at:", result.stdout.strip())
            return True
        else:
            print("✗ Ollama not found. Install with: curl -fsSL https://ollama.ai/install.sh | sh")
            return False
    except:
        return False

def test_memory_with_llm():
    """Test memory integration with LLM"""
    print("\nTesting Memory Integration with LLM...")
    print("=" * 50)
    
    # Create integration bridge
    bridge = MemoryIntegrationBridge(
        enhanced_db_path="test_integration.db",
        model_name="phi3:mini"  # Or tinyllama if phi3 not available
    )
    
    session_id = f"integration_test_{int(time.time())}"
    
    # Test 1: Store user facts with confidence
    print("\n1. Storing user information with confidence tracking...")
    
    user_inputs = [
        "My name is definitely Bob and I'm a data scientist.",
        "I think I might enjoy machine learning.",
        "I live in Boston, or maybe it was Brooklyn... not sure.",
        "I absolutely love Python programming!"
    ]
    
    for input_text in user_inputs:
        bridge.store_conversation_with_confidence(
            role="user",
            content=input_text,
            session_id=session_id,
            source_confidence=0.9  # High confidence for user input
        )
        print(f"  ✓ Stored: {input_text[:50]}...")
    
    # Test 2: Build context and query
    print("\n2. Testing context building with confidence...")
    
    queries = [
        "What is my name?",
        "What do I do for work?",
        "Where do I live?",
        "What programming language do I like?"
    ]
    
    for query in queries:
        print(f"\n  Query: {query}")
        
        # Build context with confidence
        context, context_confidence = bridge.build_context_with_confidence(
            query, session_id
        )
        
        print(f"  Context confidence: {context_confidence:.2f}")
        print(f"  Context preview: {context[:100]}..." if context else "  No context found")
        
        if context and check_ollama():
            # Query LLM with context
            prompt = f"Based on this information:\n{context}\n\nQuestion: {query}\nAnswer briefly:"
            
            response, response_confidence = bridge.query_llm_with_confidence(prompt)
            
            print(f"  Response: {response}")
            print(f"  Response confidence: {response_confidence:.2f}")
            print(f"  Combined confidence: {context_confidence * response_confidence:.2f}")
    
    # Test 3: Memory health check
    print("\n3. Enhanced Memory Health Check...")
    health = bridge.enhanced_memory.get_memory_health()
    
    print(f"  Total memories: {health['total_memories']}")
    print(f"  Average confidence: {health['average_confidence']:.2f}")
    
    # Test 4: Fact extraction test
    print("\n4. Testing fact extraction with confidence...")
    
    test_text = "I'm Alice, a software engineer who might be interested in AI."
    facts = bridge._extract_facts_with_confidence(test_text)
    
    print(f"  Input: {test_text}")
    print("  Extracted facts:")
    for fact_type, fact_value, confidence in facts:
        print(f"    - {fact_type}: {fact_value} (conf: {confidence:.2f})")
    
    return True

def test_sensor_memory_integration():
    """Test sensor to memory integration"""
    print("\n\nTesting Sensor Memory Integration...")
    print("=" * 50)
    
    from sensor_memory_integration import SensorMemoryIntegration, SensorReading
    from enhanced_memory_system import HierarchicalMemory
    
    # Create memory and sensor bridge
    memory = HierarchicalMemory("test_sensor_memory.db")
    sensor_bridge = SensorMemoryIntegration(memory)
    
    # Simulate IMU sudden movement
    print("\n1. Processing IMU sudden movement...")
    imu_reading = SensorReading(
        sensor_type='imu',
        data={
            'acceleration': [0.1, 0.2, 3.5],  # High Z acceleration
            'orientation': {'roll': 0, 'pitch': 0, 'yaw': 90}
        },
        confidence=0.9,
        timestamp=datetime.now()
    )
    
    patterns = sensor_bridge.process_sensor_input({'imu': imu_reading})
    print(f"  Extracted {len(patterns)} patterns from IMU")
    for p in patterns:
        print(f"    - {p.pattern_type}: {p.description}")
    
    # Check memory storage
    print("\n2. Checking stored sensor memories...")
    stats = sensor_bridge.get_sensor_memory_stats()
    print(f"  Total sensor memories: {stats['total_sensor_memories']}")
    for sensor, data in stats['sensor_memory_stats'].items():
        print(f"    - {sensor}: {data['count']} memories (avg conf: {data['average_confidence']:.2f})")
    
    return True

if __name__ == "__main__":
    print("Enhanced Memory System Integration Tests")
    print("=" * 70)
    
    # Check if ollama is available
    has_ollama = check_ollama()
    
    if not has_ollama:
        print("\nNote: Ollama not found. LLM tests will be skipped.")
        print("To install: curl -fsSL https://ollama.ai/install.sh | sh")
    
    # Run tests
    try:
        # Test memory with LLM (if available)
        if has_ollama:
            test_memory_with_llm()
        else:
            print("\nSkipping LLM integration tests (ollama not available)")
        
        # Test sensor integration (always runs)
        test_sensor_memory_integration()
        
        print("\n" + "=" * 70)
        print("✅ Integration tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()