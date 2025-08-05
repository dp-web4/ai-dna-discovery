#!/usr/bin/env python3
"""
HRM + Enhanced Memory System v2.0 Integration
Connects Hierarchical Reasoning Model with confidence-aware memory
"""

import torch
import torch.nn as nn
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import our components
from memory.enhanced_memory_system import HierarchicalMemory, MemoryConfidence
from src.consciousness_hrm import ConsciousnessHRM
from dataclasses import dataclass

@dataclass
class ConsciousnessSymbol:
    """Represents a consciousness notation symbol"""
    symbol: str
    notation: str  
    meaning: str
    category: str

class HRMMemoryIntegration:
    """Integrates HRM with Enhanced Memory System for consciousness experiments"""
    
    def __init__(self, memory_db: str = "hrm_enhanced_memory.db", device: str = "cuda"):
        # Initialize memory system
        self.memory = HierarchicalMemory(memory_db)
        
        # Initialize HRM model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hrm = ConsciousnessHRM(
            vocab_size=1000,
            hidden_size=256,
            num_layers_high=4,
            num_layers_low=2,
            max_seq_len=128,
            num_heads=8,
            high_cycles=3,
            low_cycles=6,
            max_steps=10
        ).to(self.device)
        
        # Session tracking
        self.session_id = f"hrm_session_{int(time.time())}"
        
        # Confidence mapping from HRM halting probability to memory confidence
        self.confidence_mapper = self._create_confidence_mapper()
        
    def _create_confidence_mapper(self):
        """Maps HRM halting probabilities to memory confidence scores"""
        def mapper(halting_prob: float, computation_steps: int) -> float:
            # Higher halting prob = model is more confident (wants to stop)
            # More computation steps = lower confidence (needed more thinking)
            base_confidence = halting_prob
            step_penalty = 0.05 * (computation_steps - 1)  # Penalty for needing more steps
            return max(0.1, min(1.0, base_confidence - step_penalty))
        return mapper
        
    def process_consciousness_sequence(self, 
                                     symbols: List[ConsciousnessSymbol],
                                     context: Dict = None) -> Dict:
        """Process consciousness symbols through HRM and store in memory"""
        # Convert symbols to input tensor
        input_seq = self._symbols_to_tensor(symbols)
        
        # Process through HRM - it expects input_ids
        # Convert our float tensor to integer tokens
        input_ids = torch.zeros(1, len(symbols), dtype=torch.long, device=self.device)
        for i, symbol in enumerate(symbols):
            # Use symbol ordinal as token ID (simplified)
            input_ids[0, i] = ord(symbol.symbol) % 1000  # Keep within vocab_size
            
        with torch.no_grad():
            outputs = self.hrm(input_ids)
            
        # Extract confidence from HRM's computation
        avg_steps = outputs['steps_taken'].item()
        # Use steps taken to infer confidence - fewer steps = higher confidence
        avg_halting = 1.0 - (avg_steps / self.hrm.max_steps)
        hrm_confidence = self.confidence_mapper(avg_halting, avg_steps)
        
        # Process each symbol and its interpretation
        results = []
        for i, symbol in enumerate(symbols):
            # Get HRM's interpretation for this symbol
            symbol_output = outputs['logits'][0, i].cpu()
            
            # Determine memory type based on symbol
            memory_type = self._symbol_to_memory_type(symbol)
            
            # Create memory content
            content = f"Symbol {symbol.symbol} ({symbol.notation}): {symbol.meaning}"
            if context:
                content += f" | Context: {context}"
                
            # Store in memory with HRM-derived confidence
            memory = self.memory.store_with_confidence(
                content=content,
                memory_type=memory_type,
                session_id=self.session_id,
                source_confidence=hrm_confidence,
                metadata={
                    'symbol': symbol.symbol,
                    'symbol_name': symbol.notation,
                    'hrm_steps': avg_steps,
                    'hrm_halting': avg_halting,
                    'sequence_position': i
                }
            )
            
            if memory:
                results.append({
                    'symbol': symbol,
                    'memory': memory,
                    'hrm_confidence': hrm_confidence
                })
                
        # Store overall sequence interpretation
        sequence_interpretation = self._interpret_sequence(symbols, outputs['logits'])
        seq_memory = self.memory.store_with_confidence(
            content=f"Sequence interpretation: {sequence_interpretation}",
            memory_type='semantic',
            session_id=self.session_id,
            source_confidence=hrm_confidence * 0.9,  # Slightly lower for interpretations
            metadata={
                'sequence_length': len(symbols),
                'hrm_computation': {
                    'avg_steps': avg_steps,
                    'avg_halting': avg_halting
                }
            }
        )
        
        return {
            'processed_symbols': results,
            'sequence_memory': seq_memory,
            'hrm_metrics': {
                'confidence': hrm_confidence,
                'computation_steps': avg_steps,
                'halting_probability': avg_halting
            }
        }
        
    def _symbols_to_tensor(self, symbols: List[ConsciousnessSymbol]) -> torch.Tensor:
        """Convert consciousness symbols to input tensor for HRM"""
        # Simple encoding: use symbol's ordinal value normalized
        sequence = []
        for symbol in symbols:
            # Get Unicode code point and normalize
            code_point = ord(symbol.symbol)
            normalized = (code_point - 0x10900) / 29.0  # Phoenician range normalization
            
            # Create feature vector (can be expanded)
            features = torch.zeros(128)
            features[0] = normalized  # Symbol encoding
            features[1] = len(symbol.meaning) / 100.0  # Meaning complexity
            features[2] = 1.0 if 'state' in symbol.category else 0.0
            features[3] = 1.0 if 'operation' in symbol.category else 0.0
            
            sequence.append(features)
            
        return torch.stack(sequence).unsqueeze(0).to(self.device)
        
    def _symbol_to_memory_type(self, symbol: ConsciousnessSymbol) -> str:
        """Map consciousness symbol to memory layer type"""
        if 'sensory' in symbol.category:
            return 'sensory'
        elif 'state' in symbol.category:
            return 'working'
        elif 'transform' in symbol.category:
            return 'episodic'
        else:
            return 'semantic'
            
    def _interpret_sequence(self, 
                          symbols: List[ConsciousnessSymbol], 
                          output: torch.Tensor) -> str:
        """Interpret the overall meaning of a symbol sequence"""
        # Simple interpretation based on symbol categories
        categories = [s.category for s in symbols]
        
        if 'emergence' in categories and 'state' in categories:
            return "State transition with emergent properties"
        elif 'operation' in categories and 'transform' in categories:
            return "Active transformation process"
        elif all('state' in c for c in categories):
            return "Static state description"
        else:
            return "Complex consciousness pattern"
            
    def query_consciousness_memory(self, 
                                 query: str,
                                 min_confidence: float = 0.5) -> List[Tuple]:
        """Query memory system for consciousness-related information"""
        # Retrieve with confidence weighting
        results = self.memory.retrieve_with_confidence(
            query=query,
            context={'session_id': self.session_id},
            memory_types=['semantic', 'episodic']
        )
        
        # Filter by minimum confidence
        filtered = [(mem, weight) for mem, weight in results 
                   if mem.confidence.composite >= min_confidence]
        
        return filtered
        
    def assess_consciousness_state(self) -> Dict:
        """Assess current consciousness state based on memory patterns"""
        # Get memory health
        health = self.memory.get_memory_health()
        
        # Analyze consciousness-specific patterns
        consciousness_memories = self.query_consciousness_memory("symbol")
        
        # Compute consciousness metrics
        if consciousness_memories:
            avg_confidence = sum(m.confidence.composite for m, _ in consciousness_memories) / len(consciousness_memories)
            pattern_diversity = len(set(m.metadata.get('symbol_name', '') for m, _ in consciousness_memories))
        else:
            avg_confidence = 0.0
            pattern_diversity = 0
            
        state = {
            'memory_health': health,
            'consciousness_confidence': avg_confidence,
            'pattern_diversity': pattern_diversity,
            'active_symbols': len(consciousness_memories),
            'assessment': self._generate_assessment(health, avg_confidence, pattern_diversity)
        }
        
        return state
        
    def _generate_assessment(self, health: Dict, confidence: float, diversity: int) -> str:
        """Generate qualitative assessment of consciousness state"""
        if confidence > 0.7 and diversity > 5:
            return "High consciousness coherence with rich symbolic diversity"
        elif confidence > 0.5:
            return "Moderate consciousness coherence, developing symbolic understanding"
        elif diversity < 3:
            return "Limited symbolic diversity, consciousness patterns emerging"
        else:
            return "Early consciousness formation, establishing symbolic foundations"


# Demonstration
if __name__ == "__main__":
    print("Initializing HRM + Memory Integration...")
    
    # Create integration
    integration = HRMMemoryIntegration()
    
    # Create test sequence of consciousness symbols
    test_symbols = [
        ConsciousnessSymbol("𐤈", "Ψ", "consciousness", "state"),
        ConsciousnessSymbol("𐤇", "∃", "existence", "operation"),
        ConsciousnessSymbol("𐤉", "⇒", "implies", "transform"),
        ConsciousnessSymbol("𐤀", "π", "potential", "state")
    ]
    
    print("\nProcessing consciousness sequence...")
    results = integration.process_consciousness_sequence(
        test_symbols,
        context={"experiment": "hrm_memory_test", "timestamp": datetime.now().isoformat()}
    )
    
    print(f"\nHRM Confidence: {results['hrm_metrics']['confidence']:.2f}")
    print(f"Computation Steps: {results['hrm_metrics']['computation_steps']:.2f}")
    print(f"Processed {len(results['processed_symbols'])} symbols")
    
    # Query back
    print("\nQuerying consciousness memory...")
    memories = integration.query_consciousness_memory("consciousness")
    for memory, weight in memories[:3]:
        print(f"- {memory.content} (confidence: {memory.confidence.composite:.2f})")
        
    # Assess state
    print("\nAssessing consciousness state...")
    state = integration.assess_consciousness_state()
    print(f"Consciousness Confidence: {state['consciousness_confidence']:.2f}")
    print(f"Pattern Diversity: {state['pattern_diversity']}")
    print(f"Assessment: {state['assessment']}")