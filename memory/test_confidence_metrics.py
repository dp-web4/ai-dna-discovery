#!/usr/bin/env python3
"""
Test harness for memory confidence metrics
Tests confidence scoring, hierarchical layers, and retrieval accuracy
"""

import json
import time
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_memory_system import HierarchicalMemory, MemoryConfidence, Memory

class MemoryConfidenceTestHarness:
    """Comprehensive test suite for memory confidence metrics"""
    
    def __init__(self, db_path: str = "test_confidence_metrics.db"):
        self.memory_system = HierarchicalMemory(db_path)
        self.test_results = {
            'confidence_accuracy': [],
            'retrieval_precision': [],
            'temporal_decay': [],
            'layer_distribution': {}
        }
        
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("Starting Memory Confidence Test Suite...")
        print("=" * 50)
        
        # Test 1: Confidence scoring accuracy
        self.test_confidence_scoring()
        
        # Test 2: Retrieval precision
        self.test_retrieval_precision()
        
        # Test 3: Temporal decay effects
        self.test_temporal_decay()
        
        # Test 4: Hierarchical layer distribution
        self.test_layer_distribution()
        
        # Test 5: Memory consolidation
        self.test_memory_consolidation()
        
        # Test 6: Edge cases
        self.test_edge_cases()
        
        # Generate report
        self.generate_test_report()
        
    def test_confidence_scoring(self):
        """Test confidence scoring accuracy"""
        print("\n1. Testing Confidence Scoring...")
        
        test_memories = [
            # (content, expected_confidence_range, source_confidence)
            ("User name is definitely Bob", (0.8, 1.0), 0.95),
            ("User might be interested in AI", (0.5, 0.7), 0.6),
            ("User possibly likes coffee", (0.3, 0.5), 0.4),
            ("Uncertain about user location", (0.1, 0.3), 0.2),
        ]
        
        session_id = f"confidence_test_{int(time.time())}"
        results = []
        
        for content, expected_range, source_conf in test_memories:
            memory = self.memory_system.store_with_confidence(
                content, "semantic", session_id, source_conf
            )
            
            if memory:
                actual_conf = memory.confidence.composite
                in_range = expected_range[0] <= actual_conf <= expected_range[1]
                results.append({
                    'content': content,
                    'expected': expected_range,
                    'actual': actual_conf,
                    'correct': in_range
                })
                print(f"  ✓ {content[:30]}... - Confidence: {actual_conf:.2f} {'✓' if in_range else '✗'}")
            else:
                print(f"  ✗ {content[:30]}... - Rejected (too low confidence)")
                
        accuracy = sum(r['correct'] for r in results if 'correct' in r) / len(test_memories)
        self.test_results['confidence_accuracy'] = accuracy
        print(f"\nConfidence Scoring Accuracy: {accuracy:.1%}")
        
    def test_retrieval_precision(self):
        """Test retrieval precision with confidence weighting"""
        print("\n2. Testing Retrieval Precision...")
        
        session_id = f"retrieval_test_{int(time.time())}"
        
        # Store test memories with varying confidence
        test_data = [
            ("Alice is a software engineer", 0.9, {'concept': 'profession'}),
            ("Alice enjoys programming", 0.8, {'concept': 'interest'}),
            ("Alice might like Python", 0.6, {'concept': 'skill'}),
            ("Bob is a data scientist", 0.9, {'concept': 'profession'}),
            ("Carol works in marketing", 0.7, {'concept': 'profession'}),
        ]
        
        for content, conf, metadata in test_data:
            self.memory_system.store_with_confidence(
                content, "semantic", session_id, conf, metadata
            )
            
        # Test retrieval queries
        queries = [
            ("Alice", ["Alice is a software engineer", "Alice enjoys programming"]),
            ("profession", ["Alice is a software engineer", "Bob is a data scientist"]),
            ("programming", ["Alice enjoys programming", "Alice might like Python"]),
        ]
        
        total_precision = 0
        for query, expected_top in queries:
            results = self.memory_system.retrieve_with_confidence(query, limit=2)
            retrieved = [mem.content for mem, _ in results[:2]]
            
            # Calculate precision
            correct = sum(1 for r in retrieved if r in expected_top)
            precision = correct / len(expected_top) if expected_top else 0
            total_precision += precision
            
            print(f"  Query: '{query}' - Precision: {precision:.1%}")
            print(f"    Expected: {expected_top}")
            print(f"    Retrieved: {retrieved}")
            
        avg_precision = total_precision / len(queries)
        self.test_results['retrieval_precision'] = avg_precision
        print(f"\nAverage Retrieval Precision: {avg_precision:.1%}")
        
    def test_temporal_decay(self):
        """Test temporal decay effects on memory relevance"""
        print("\n3. Testing Temporal Decay...")
        
        session_id = f"decay_test_{int(time.time())}"
        
        # Create memories at different time points
        memories = []
        base_time = datetime.now()
        
        for days_ago in [0, 1, 7, 30, 90]:
            # Manually create memory with past timestamp
            memory = Memory(
                content=f"Event from {days_ago} days ago",
                memory_type="episodic",
                timestamp=base_time - timedelta(days=days_ago),
                confidence=MemoryConfidence(0.8, 0.8, 0.8, 0.8),
                session_id=session_id
            )
            memories.append((memory, days_ago))
            
        # Test decay calculation
        decay_results = []
        for memory, days_ago in memories:
            decay_factor = self.memory_system._temporal_decay(memory.timestamp)
            decay_results.append({
                'days_ago': days_ago,
                'decay_factor': decay_factor
            })
            print(f"  {days_ago} days ago: decay factor = {decay_factor:.3f}")
            
        self.test_results['temporal_decay'] = decay_results
        
        # Verify exponential decay
        decay_factors = [r['decay_factor'] for r in decay_results]
        is_decreasing = all(decay_factors[i] >= decay_factors[i+1] 
                          for i in range(len(decay_factors)-1))
        print(f"\nTemporal decay is properly decreasing: {'✓' if is_decreasing else '✗'}")
        
    def test_layer_distribution(self):
        """Test distribution across hierarchical memory layers"""
        print("\n4. Testing Hierarchical Layer Distribution...")
        
        session_id = f"layer_test_{int(time.time())}"
        
        # Store memories in different layers
        layer_tests = [
            ("sensory", "Quick visual flash", 0.5),
            ("working", "Current task context", 0.7),
            ("episodic", "Meeting with team yesterday", 0.8),
            ("semantic", "Python is a programming language", 0.9),
        ]
        
        for layer, content, conf in layer_tests:
            self.memory_system.store_with_confidence(
                content, layer, session_id, conf
            )
            
        # Check layer populations
        layer_counts = {
            'sensory': len(self.memory_system.sensory_buffer),
            'working': len(self.memory_system.working_memory),
            'episodic': len(self.memory_system.episodic_memory),
            'semantic': sum(len(v) for v in self.memory_system.semantic_memory.values())
        }
        
        self.test_results['layer_distribution'] = layer_counts
        
        print("\nLayer Distribution:")
        for layer, count in layer_counts.items():
            print(f"  {layer}: {count} memories")
            
    def test_memory_consolidation(self):
        """Test memory consolidation from short-term to long-term"""
        print("\n5. Testing Memory Consolidation...")
        
        session_id = f"consolidation_test_{int(time.time())}"
        
        # Fill working memory with various confidence levels
        for i in range(10):
            conf = 0.5 + (i * 0.05)  # 0.5 to 0.95
            self.memory_system.store_with_confidence(
                f"Working memory item {i}",
                "working",
                session_id,
                conf
            )
            
        initial_working = len(self.memory_system.working_memory)
        
        # Trigger consolidation
        self.memory_system.consolidate_memories()
        
        final_working = len(self.memory_system.working_memory)
        
        print(f"  Working memory before: {initial_working}")
        print(f"  Working memory after: {final_working}")
        print(f"  Memories consolidated: {initial_working - final_working}")
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n6. Testing Edge Cases...")
        
        session_id = f"edge_test_{int(time.time())}"
        
        # Test 1: Empty content
        result1 = self.memory_system.store_with_confidence(
            "", "semantic", session_id, 0.9
        )
        print(f"  Empty content: {'Rejected' if not result1 else 'Accepted'}")
        
        # Test 2: Very long content
        long_content = "x" * 10000
        result2 = self.memory_system.store_with_confidence(
            long_content, "episodic", session_id, 0.8
        )
        print(f"  Very long content: {'Stored' if result2 else 'Failed'}")
        
        # Test 3: Invalid confidence values
        try:
            self.memory_system.store_with_confidence(
                "Test", "semantic", session_id, 1.5  # Invalid: > 1.0
            )
            print(f"  Invalid confidence: Not caught (bug!)")
        except:
            print(f"  Invalid confidence: Properly handled")
            
        # Test 4: Retrieve from empty database
        empty_memory = HierarchicalMemory("empty_test.db")
        results = empty_memory.retrieve_with_confidence("test")
        print(f"  Empty database retrieval: {len(results)} results")
        
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 50)
        print("MEMORY CONFIDENCE TEST REPORT")
        print("=" * 50)
        
        # Overall statistics
        print("\nTest Results Summary:")
        print(f"  Confidence Accuracy: {self.test_results.get('confidence_accuracy', 0):.1%}")
        print(f"  Retrieval Precision: {self.test_results.get('retrieval_precision', 0):.1%}")
        
        # Memory health check
        health = self.memory_system.get_memory_health()
        print(f"\nMemory System Health:")
        print(f"  Total Memories: {health['total_memories']}")
        print(f"  Average Confidence: {health['average_confidence']:.2f}")
        print(f"  Working Memory Load: {health['working_memory_load']:.1%}")
        
        if health['recommendations']:
            print("\nRecommendations:")
            for rec in health['recommendations']:
                print(f"  - {rec}")
                
        # Save detailed results
        with open('memory_confidence_test_results.json', 'w') as f:
            json.dump({
                'test_results': self.test_results,
                'memory_health': health,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
            
        print("\nDetailed results saved to: memory_confidence_test_results.json")
        
    def visualize_confidence_distribution(self):
        """Create visualization of confidence distribution"""
        conn = sqlite3.connect(self.memory_system.db_path)
        c = conn.cursor()
        
        # Get confidence scores
        c.execute('SELECT confidence_score FROM memory_confidence')
        scores = [row[0] for row in c.fetchall()]
        conn.close()
        
        if scores:
            plt.figure(figsize=(10, 6))
            
            # Histogram
            plt.subplot(1, 2, 1)
            plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            plt.title('Confidence Score Distribution')
            
            # Box plot by memory type
            plt.subplot(1, 2, 2)
            conn = sqlite3.connect(self.memory_system.db_path)
            c = conn.cursor()
            c.execute('SELECT memory_type, confidence_score FROM memory_confidence')
            data = c.fetchall()
            conn.close()
            
            if data:
                import pandas as pd
                df = pd.DataFrame(data, columns=['Type', 'Confidence'])
                df.boxplot(column='Confidence', by='Type')
                plt.title('Confidence by Memory Type')
                
            plt.tight_layout()
            plt.savefig('memory_confidence_distribution.png')
            print("\nConfidence distribution plot saved to: memory_confidence_distribution.png")

# Run tests if executed directly
if __name__ == "__main__":
    harness = MemoryConfidenceTestHarness()
    harness.run_all_tests()
    
    # Optional: visualize results (requires matplotlib)
    try:
        harness.visualize_confidence_distribution()
    except ImportError:
        print("\nVisualization skipped (matplotlib not available)")
    except Exception as e:
        print(f"\nVisualization error: {e}")