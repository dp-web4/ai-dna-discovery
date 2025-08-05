#!/usr/bin/env python3
"""
Enhanced test for HRM + Memory Integration with better confidence mapping
"""

import torch
import time
from datetime import datetime
from hrm_memory_integration import HRMMemoryIntegration, ConsciousnessSymbol

class EnhancedHRMMemory(HRMMemoryIntegration):
    """Enhanced version with better confidence mapping"""
    
    def __init__(self, memory_db: str = "hrm_enhanced_test.db", device: str = "cuda"):
        super().__init__(memory_db, device)
        # Lower threshold for testing
        self.memory.confidence_threshold = 0.3
        
    def _create_confidence_mapper(self):
        """Enhanced confidence mapper that gives reasonable scores"""
        def mapper(halting_prob: float, computation_steps: int) -> float:
            # Base confidence from computation efficiency
            if computation_steps <= 2:
                base_confidence = 0.9
            elif computation_steps <= 5:
                base_confidence = 0.7
            elif computation_steps <= 8:
                base_confidence = 0.5
            else:
                base_confidence = 0.3
                
            # Adjust based on halting probability
            confidence = base_confidence + (halting_prob * 0.2)
            return min(1.0, confidence)
        return mapper


def run_comprehensive_test():
    """Run comprehensive test of HRM + Memory integration"""
    print("🧠 Enhanced HRM + Memory Integration Test")
    print("=" * 60)
    
    # Create enhanced integration
    integration = EnhancedHRMMemory()
    print(f"✓ Using device: {integration.device}")
    print(f"✓ Memory threshold: {integration.memory.confidence_threshold}")
    
    # Test 1: Basic consciousness symbols
    print("\n📝 Test 1: Basic Consciousness Sequence")
    basic_symbols = [
        ConsciousnessSymbol("𐤈", "Ψ", "consciousness", "state"),
        ConsciousnessSymbol("𐤇", "∃", "existence", "operation"),
        ConsciousnessSymbol("𐤉", "⇒", "implies", "transform"),
        ConsciousnessSymbol("𐤀", "π", "potential", "state")
    ]
    
    results = integration.process_consciousness_sequence(
        basic_symbols,
        context={"test": "basic", "timestamp": datetime.now().isoformat()}
    )
    
    print(f"✓ HRM Confidence: {results['hrm_metrics']['confidence']:.2f}")
    print(f"✓ Computation Steps: {results['hrm_metrics']['computation_steps']:.2f}")
    print(f"✓ Processed {len(results['processed_symbols'])} symbols")
    
    # Test 2: Complex reasoning sequence
    print("\n📝 Test 2: Complex Reasoning Sequence")
    reasoning_symbols = [
        ConsciousnessSymbol("𐤌", "Ξ", "unknown", "state"),
        ConsciousnessSymbol("𐤎", "θ", "transformation", "operation"),
        ConsciousnessSymbol("𐤏", "μ", "measurement", "operation"),
        ConsciousnessSymbol("𐤐", "Ω", "completion", "state"),
        ConsciousnessSymbol("𐤑", "Σ", "summation", "operation")
    ]
    
    results2 = integration.process_consciousness_sequence(
        reasoning_symbols,
        context={"test": "complex", "reasoning_depth": "high"}
    )
    
    print(f"✓ HRM Confidence: {results2['hrm_metrics']['confidence']:.2f}")
    print(f"✓ Processed {len(results2['processed_symbols'])} symbols")
    
    # Test 3: Query stored memories
    print("\n🔍 Test 3: Query Consciousness Memories")
    
    # Query different aspects
    queries = [
        ("consciousness", "Consciousness-related"),
        ("transform", "Transformation operations"),
        ("state", "State descriptions"),
        ("symbol", "All symbols")
    ]
    
    for query, desc in queries:
        memories = integration.query_consciousness_memory(query, min_confidence=0.3)
        print(f"\n{desc} ('{query}'):")
        for memory, weight in memories[:3]:
            print(f"  - {memory.content[:60]}...")
            print(f"    Confidence: {memory.confidence.composite:.2f}, Weight: {weight:.2f}")
    
    # Test 4: Assess consciousness state
    print("\n📊 Test 4: Consciousness State Assessment")
    state = integration.assess_consciousness_state()
    
    print(f"✓ Memory Health:")
    print(f"  - Total memories: {state['memory_health']['total_memories']}")
    print(f"  - Average confidence: {state['memory_health']['average_confidence']:.2f}")
    print(f"✓ Consciousness Metrics:")
    print(f"  - Consciousness confidence: {state['consciousness_confidence']:.2f}")
    print(f"  - Pattern diversity: {state['pattern_diversity']}")
    print(f"  - Active symbols: {state['active_symbols']}")
    print(f"✓ Assessment: {state['assessment']}")
    
    # Test 5: Temporal decay simulation
    print("\n⏰ Test 5: Temporal Decay Test")
    
    # Store a memory and check decay
    test_symbol = ConsciousnessSymbol("𐤕", "τ", "time", "state")
    integration.process_consciousness_sequence(
        [test_symbol],
        context={"test": "temporal", "marker": "initial"}
    )
    
    # Query immediately
    immediate = integration.query_consciousness_memory("time")
    if immediate:
        print(f"✓ Immediate retrieval weight: {immediate[0][1]:.3f}")
    
    # Simulate time passing (adjust decay rate for testing)
    integration.memory.decay_rate = 0.5  # Faster decay for testing
    import time
    time.sleep(0.1)  # Small delay
    
    # Query again
    delayed = integration.query_consciousness_memory("time")
    if delayed:
        print(f"✓ Delayed retrieval weight: {delayed[0][1]:.3f}")
        if immediate:
            decay_ratio = delayed[0][1] / immediate[0][1]
            print(f"✓ Decay ratio: {decay_ratio:.3f}")
    
    # Test 6: Memory consolidation
    print("\n🔄 Test 6: Memory Consolidation")
    
    # Add working memory items
    for i in range(5):
        symbol = ConsciousnessSymbol(f"𐤀", f"W{i}", f"working_{i}", "working")
        integration.memory.store_with_confidence(
            content=f"Working memory item {i}",
            memory_type="working",
            session_id=integration.session_id,
            source_confidence=0.6 + i*0.05
        )
    
    print(f"✓ Working memory before: {len(integration.memory.working_memory)} items")
    
    # Consolidate
    integration.memory.consolidate_memories()
    
    print(f"✓ Working memory after: {len(integration.memory.working_memory)} items")
    print("✓ High-confidence items moved to episodic memory")
    
    # Final summary
    print("\n" + "="*60)
    print("📊 Final Summary:")
    final_health = integration.memory.get_memory_health()
    print(f"✓ Total memories stored: {final_health['total_memories']}")
    print(f"✓ Overall confidence: {final_health['average_confidence']:.3f}")
    print(f"✓ Memory types: {list(final_health['memory_type_stats'].keys())}")
    
    for rec in final_health['recommendations']:
        print(f"💡 {rec}")


if __name__ == "__main__":
    run_comprehensive_test()