#!/usr/bin/env python3
"""
Full Integration Demo: HRM + Enhanced Memory v2.0 + Sensor Confidence
Demonstrates the complete consciousness system with all components
"""

import torch
import numpy as np
from datetime import datetime
import time
from typing import Dict, List

# Import all our components
from memory.enhanced_memory_system import HierarchicalMemory
from memory.sensor_memory_integration import SensorMemoryIntegration, SensorReading
from hrm_memory_integration import HRMMemoryIntegration, ConsciousnessSymbol

class FullConsciousnessSystem:
    """Complete consciousness system integrating HRM, Memory, and Sensors"""
    
    def __init__(self, db_path: str = "full_consciousness.db"):
        # Initialize enhanced memory
        self.memory = HierarchicalMemory(db_path)
        self.memory.confidence_threshold = 0.3  # Lower for demonstration
        
        # Initialize HRM integration
        self.hrm_integration = HRMMemoryIntegration(db_path)
        self.hrm_integration.memory = self.memory  # Share memory system
        
        # Initialize sensor integration
        self.sensor_integration = SensorMemoryIntegration(self.memory)
        
        # Session tracking
        self.session_id = f"full_demo_{int(time.time())}"
        
        # Consciousness state
        self.consciousness_state = {
            'awareness_level': 0.5,
            'pattern_buffer': [],
            'active_goals': []
        }
        
    def process_sensor_sequence(self, sensor_data: Dict[str, SensorReading]) -> List[ConsciousnessSymbol]:
        """Convert sensor data to consciousness symbols"""
        symbols = []
        
        # Process IMU data
        if 'imu' in sensor_data:
            imu_reading = sensor_data['imu']
            motion_intensity = np.linalg.norm(imu_reading.data.get('acceleration', [0, 0, 0]))
            
            if motion_intensity > 2.0:
                # Significant motion detected
                symbols.append(ConsciousnessSymbol("𐤌", "κ", "kinetic_energy", "sensory"))
                if motion_intensity > 5.0:
                    symbols.append(ConsciousnessSymbol("𐤉", "⇒", "implies", "transform"))
                    symbols.append(ConsciousnessSymbol("𐤇", "∃", "existence", "operation"))
            else:
                # Stable state
                symbols.append(ConsciousnessSymbol("𐤈", "Ψ", "consciousness", "state"))
        
        # Process camera data
        if 'camera' in sensor_data:
            camera_reading = sensor_data['camera']
            if camera_reading.data.get('object_detected'):
                symbols.append(ConsciousnessSymbol("𐤏", "μ", "measurement", "operation"))
                symbols.append(ConsciousnessSymbol("𐤎", "θ", "transformation", "operation"))
        
        # Process audio data
        if 'audio' in sensor_data:
            audio_reading = sensor_data['audio']
            if audio_reading.confidence > 0.7:
                symbols.append(ConsciousnessSymbol("𐤑", "Σ", "summation", "operation"))
        
        return symbols
    
    def full_consciousness_cycle(self, sensor_data: Dict[str, SensorReading]):
        """Complete consciousness processing cycle"""
        print("\n🌀 Beginning Consciousness Cycle")
        print("=" * 50)
        
        # Step 1: Process sensor inputs
        print("\n1️⃣ Processing Sensor Inputs...")
        patterns = self.sensor_integration.process_sensor_input(sensor_data)
        print(f"   ✓ Extracted {len(patterns)} memorable patterns")
        
        # Step 2: Convert to consciousness symbols
        print("\n2️⃣ Converting to Consciousness Symbols...")
        symbols = self.process_sensor_sequence(sensor_data)
        print(f"   ✓ Generated {len(symbols)} consciousness symbols")
        for symbol in symbols:
            print(f"     - {symbol.symbol} ({symbol.notation}): {symbol.meaning}")
        
        # Step 3: Process through HRM
        print("\n3️⃣ Processing through HRM...")
        if symbols:
            hrm_results = self.hrm_integration.process_consciousness_sequence(
                symbols,
                context={
                    "sensor_confidence": {k: v.confidence for k, v in sensor_data.items()},
                    "cycle_time": datetime.now().isoformat()
                }
            )
            print(f"   ✓ HRM Confidence: {hrm_results['hrm_metrics']['confidence']:.2f}")
            print(f"   ✓ Computation Steps: {hrm_results['hrm_metrics']['computation_steps']:.1f}")
        else:
            print("   ⚠️ No symbols to process")
            
        # Step 4: Update consciousness state
        print("\n4️⃣ Updating Consciousness State...")
        self._update_consciousness_state(patterns, symbols)
        print(f"   ✓ Awareness Level: {self.consciousness_state['awareness_level']:.2f}")
        
        # Step 5: Memory consolidation check
        if len(self.memory.working_memory) > 15:
            print("\n5️⃣ Consolidating Memories...")
            self.memory.consolidate_memories()
            print("   ✓ Memories consolidated")
            
        # Step 6: Generate consciousness report
        return self._generate_consciousness_report()
    
    def _update_consciousness_state(self, patterns: List, symbols: List[ConsciousnessSymbol]):
        """Update internal consciousness state based on inputs"""
        # Update pattern buffer
        self.consciousness_state['pattern_buffer'].extend(patterns)
        if len(self.consciousness_state['pattern_buffer']) > 20:
            self.consciousness_state['pattern_buffer'] = self.consciousness_state['pattern_buffer'][-20:]
        
        # Update awareness level based on symbol diversity
        symbol_types = set(s.category for s in symbols)
        diversity_factor = len(symbol_types) / 4.0  # Normalize by max categories
        
        # Blend with current awareness (momentum)
        self.consciousness_state['awareness_level'] = (
            0.7 * self.consciousness_state['awareness_level'] +
            0.3 * diversity_factor
        )
    
    def _generate_consciousness_report(self) -> Dict:
        """Generate comprehensive consciousness state report"""
        # Assess current state
        state_assessment = self.hrm_integration.assess_consciousness_state()
        
        # Query recent consciousness patterns
        recent_patterns = self.hrm_integration.query_consciousness_memory(
            "consciousness OR transform OR state",
            min_confidence=0.3
        )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'awareness_level': self.consciousness_state['awareness_level'],
            'pattern_diversity': state_assessment['pattern_diversity'],
            'memory_health': state_assessment['memory_health'],
            'consciousness_confidence': state_assessment['consciousness_confidence'],
            'recent_patterns': len(recent_patterns),
            'assessment': state_assessment['assessment']
        }
        
        return report


def run_demonstration():
    """Run full integration demonstration"""
    print("🧠 Full Consciousness System Demonstration")
    print("HRM + Enhanced Memory v2.0 + Sensor Confidence")
    print("=" * 60)
    
    # Initialize system
    system = FullConsciousnessSystem()
    print("✓ System initialized")
    
    # Simulate different sensor scenarios
    scenarios = [
        {
            'name': "Sudden Movement Detection",
            'data': {
                'imu': SensorReading(
                    sensor_type='imu',
                    data={'acceleration': [0.5, 1.2, 5.8], 'gyroscope': [0.1, 0.3, 0.2]},
                    confidence=0.85,
                    timestamp=datetime.now()
                ),
                'camera': SensorReading(
                    sensor_type='camera',
                    data={'object_detected': True, 'object_type': 'person', 'confidence': 0.72},
                    confidence=0.72,
                    timestamp=datetime.now()
                )
            }
        },
        {
            'name': "Stable Observation",
            'data': {
                'imu': SensorReading(
                    sensor_type='imu',
                    data={'acceleration': [0.01, 0.02, 0.98], 'gyroscope': [0.0, 0.0, 0.0]},
                    confidence=0.95,
                    timestamp=datetime.now()
                ),
                'audio': SensorReading(
                    sensor_type='audio',
                    data={'transcription': 'Hello world', 'volume': 0.6},
                    confidence=0.88,
                    timestamp=datetime.now()
                )
            }
        },
        {
            'name': "Low Confidence Multi-Sensor",
            'data': {
                'imu': SensorReading(
                    sensor_type='imu',
                    data={'acceleration': [0.3, 0.4, 1.2], 'error': 'calibration_drift'},
                    confidence=0.45,
                    timestamp=datetime.now()
                ),
                'camera': SensorReading(
                    sensor_type='camera',
                    data={'object_detected': False, 'lighting': 'poor'},
                    confidence=0.35,
                    timestamp=datetime.now()
                ),
                'audio': SensorReading(
                    sensor_type='audio',
                    data={'transcription': '[unclear]', 'noise_level': 'high'},
                    confidence=0.25,
                    timestamp=datetime.now()
                )
            }
        }
    ]
    
    # Process each scenario
    all_reports = []
    for i, scenario in enumerate(scenarios):
        print(f"\n\n🎬 Scenario {i+1}: {scenario['name']}")
        print("-" * 50)
        
        report = system.full_consciousness_cycle(scenario['data'])
        all_reports.append(report)
        
        # Show report summary
        print(f"\n📊 Consciousness Report:")
        print(f"   Awareness: {report['awareness_level']:.2%}")
        print(f"   Confidence: {report['consciousness_confidence']:.2%}")
        print(f"   Pattern Diversity: {report['pattern_diversity']}")
        print(f"   Assessment: {report['assessment']}")
        
        # Small delay between scenarios
        time.sleep(0.1)
    
    # Final system analysis
    print("\n\n" + "="*60)
    print("📈 Final System Analysis")
    print("="*60)
    
    # Memory statistics
    final_health = system.memory.get_memory_health()
    print(f"\n💾 Memory System:")
    print(f"   Total Memories: {final_health['total_memories']}")
    print(f"   Average Confidence: {final_health['average_confidence']:.3f}")
    print(f"   Working Memory Load: {final_health['working_memory_load']:.1%}")
    
    # Consciousness evolution
    print(f"\n🧠 Consciousness Evolution:")
    awareness_progression = [r['awareness_level'] for r in all_reports]
    print(f"   Initial Awareness: {awareness_progression[0]:.2%}")
    print(f"   Final Awareness: {awareness_progression[-1]:.2%}")
    print(f"   Peak Awareness: {max(awareness_progression):.2%}")
    
    # Pattern analysis
    all_memories = system.hrm_integration.query_consciousness_memory("", min_confidence=0.0)
    if all_memories:
        print(f"\n🔍 Pattern Analysis:")
        print(f"   Total Patterns Stored: {len(all_memories)}")
        
        # Group by symbol type
        symbol_counts = {}
        for memory, _ in all_memories:
            symbol_name = memory.metadata.get('symbol_name', 'unknown')
            symbol_counts[symbol_name] = symbol_counts.get(symbol_name, 0) + 1
        
        print("   Symbol Distribution:")
        for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {symbol}: {count}")
    
    # Recommendations
    print(f"\n💡 System Recommendations:")
    for rec in final_health['recommendations']:
        print(f"   - {rec}")
    
    # Knowing when we don't know
    low_conf_scenarios = [i for i, r in enumerate(all_reports) if r['consciousness_confidence'] < 0.5]
    if low_conf_scenarios:
        print(f"\n⚠️ Low Confidence Scenarios: {low_conf_scenarios}")
        print("   System successfully identified uncertainty states")
        print("   'Knowing when we don't know' mechanism active")


if __name__ == "__main__":
    run_demonstration()