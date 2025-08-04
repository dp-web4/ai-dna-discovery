#!/usr/bin/env python3
"""
Integration between sensor confidence framework and memory system
Bridges physical sensors (IMU, camera, audio) with memory confidence
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import sqlite3

from enhanced_memory_system import HierarchicalMemory, MemoryConfidence

# Try to import numpy, but make it optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Unified sensor reading with confidence"""
    sensor_type: str  # 'imu', 'camera', 'audio', etc.
    data: Dict
    confidence: float
    timestamp: datetime
    metadata: Dict = None
    
@dataclass
class MemorablePattern:
    """Pattern extracted from sensor data worth remembering"""
    pattern_type: str
    description: str
    confidence: float
    sensor_type: str
    is_immediate: bool  # For sensory memory
    is_significant: bool  # For episodic memory
    metadata: Dict = None

class SensorMemoryIntegration:
    """Bridge sensor confidence to memory confidence"""
    
    def __init__(self, memory_system: HierarchicalMemory):
        self.memory_system = memory_system
        
        # Pattern detection thresholds
        self.motion_threshold = 0.5  # For significant motion
        self.visual_saliency_threshold = 0.6  # For memorable visual patterns
        self.audio_significance_threshold = 0.7  # For important sounds
        
        # Sensor confidence weights for memory
        self.sensor_weights = {
            'imu': 0.8,      # IMU is generally reliable
            'camera': 0.9,   # Visual data is highly informative
            'audio': 0.7,    # Audio can be noisy
            'bluetooth': 0.6  # Bluetooth connections are binary but timing matters
        }
        
    def process_sensor_input(self, sensor_data: Dict[str, SensorReading]) -> List[MemorablePattern]:
        """Process raw sensor data and store memorable patterns"""
        patterns = []
        
        for sensor_type, reading in sensor_data.items():
            # Extract patterns based on sensor type
            if sensor_type == 'imu':
                patterns.extend(self._process_imu_data(reading))
            elif sensor_type == 'camera':
                patterns.extend(self._process_camera_data(reading))
            elif sensor_type == 'audio':
                patterns.extend(self._process_audio_data(reading))
            elif sensor_type == 'bluetooth':
                patterns.extend(self._process_bluetooth_data(reading))
                
        # Store patterns in appropriate memory layers
        for pattern in patterns:
            self._store_pattern_as_memory(pattern)
            
        return patterns
        
    def _process_imu_data(self, reading: SensorReading) -> List[MemorablePattern]:
        """Extract memorable patterns from IMU data"""
        patterns = []
        data = reading.data
        
        # Check for significant motion events
        if 'acceleration' in data:
            if HAS_NUMPY:
                accel_magnitude = np.linalg.norm(data['acceleration'])
            else:
                # Manual magnitude calculation without numpy
                accel_magnitude = (sum(x**2 for x in data['acceleration'])) ** 0.5
            
            # Sudden movement detection
            if accel_magnitude > 2.0:  # 2g threshold
                patterns.append(MemorablePattern(
                    pattern_type='sudden_movement',
                    description=f"Sudden movement detected: {accel_magnitude:.1f}g",
                    confidence=reading.confidence * 0.9,
                    sensor_type='imu',
                    is_immediate=True,
                    is_significant=True,
                    metadata={'magnitude': accel_magnitude}
                ))
                
        # Orientation changes
        if 'orientation' in data:
            # Check for significant orientation changes (would need history)
            patterns.append(MemorablePattern(
                pattern_type='orientation',
                description=f"Current orientation: {data['orientation']}",
                confidence=reading.confidence * 0.7,
                sensor_type='imu',
                is_immediate=True,
                is_significant=False
            ))
            
        return patterns
        
    def _process_camera_data(self, reading: SensorReading) -> List[MemorablePattern]:
        """Extract memorable patterns from camera data"""
        patterns = []
        data = reading.data
        
        # Object detection
        if 'detected_objects' in data:
            for obj in data['detected_objects']:
                if obj.get('confidence', 0) > self.visual_saliency_threshold:
                    patterns.append(MemorablePattern(
                        pattern_type='visual_object',
                        description=f"Detected {obj['class']}: {obj.get('attributes', '')}",
                        confidence=reading.confidence * obj['confidence'],
                        sensor_type='camera',
                        is_immediate=False,
                        is_significant=True,
                        metadata=obj
                    ))
                    
        # Scene changes
        if 'scene_change' in data and data['scene_change'] > 0.5:
            patterns.append(MemorablePattern(
                pattern_type='scene_change',
                description="Significant scene change detected",
                confidence=reading.confidence * 0.8,
                sensor_type='camera',
                is_immediate=True,
                is_significant=True
            ))
            
        return patterns
        
    def _process_audio_data(self, reading: SensorReading) -> List[MemorablePattern]:
        """Extract memorable patterns from audio data"""
        patterns = []
        data = reading.data
        
        # Speech recognition
        if 'transcription' in data:
            patterns.append(MemorablePattern(
                pattern_type='speech',
                description=f"Heard: '{data['transcription']}'",
                confidence=reading.confidence * data.get('transcription_confidence', 0.7),
                sensor_type='audio',
                is_immediate=False,
                is_significant=True,
                metadata={'speaker': data.get('speaker', 'unknown')}
            ))
            
        # Sound events
        if 'sound_events' in data:
            for event in data['sound_events']:
                if event['confidence'] > self.audio_significance_threshold:
                    patterns.append(MemorablePattern(
                        pattern_type='sound_event',
                        description=f"Sound event: {event['type']}",
                        confidence=reading.confidence * event['confidence'],
                        sensor_type='audio',
                        is_immediate=True,
                        is_significant=event.get('is_important', False)
                    ))
                    
        return patterns
        
    def _process_bluetooth_data(self, reading: SensorReading) -> List[MemorablePattern]:
        """Extract memorable patterns from Bluetooth data"""
        patterns = []
        data = reading.data
        
        # Device connections
        if 'connected_device' in data:
            patterns.append(MemorablePattern(
                pattern_type='device_connection',
                description=f"Connected to: {data['connected_device']}",
                confidence=reading.confidence,
                sensor_type='bluetooth',
                is_immediate=False,
                is_significant=True,
                metadata={'device_id': data.get('device_id')}
            ))
            
        return patterns
        
    def _store_pattern_as_memory(self, pattern: MemorablePattern):
        """Store memorable pattern in appropriate memory layer"""
        # Calculate memory confidence from sensor confidence
        memory_confidence = self._calculate_memory_confidence(pattern)
        
        # Determine memory type based on pattern characteristics
        if pattern.is_immediate and not pattern.is_significant:
            memory_type = 'sensory'
        elif pattern.is_significant:
            memory_type = 'episodic'
        else:
            memory_type = 'working'
            
        # Create session ID from timestamp
        session_id = f"sensor_{pattern.sensor_type}_{int(datetime.now().timestamp())}"
        
        # Store in memory system
        self.memory_system.store_with_confidence(
            content=pattern.description,
            memory_type=memory_type,
            session_id=session_id,
            source_confidence=memory_confidence,
            metadata={
                'pattern_type': pattern.pattern_type,
                'sensor_type': pattern.sensor_type,
                'sensor_metadata': pattern.metadata or {}
            }
        )
        
        logger.info(f"Stored {pattern.pattern_type} as {memory_type} memory (conf: {memory_confidence:.2f})")
        
    def _calculate_memory_confidence(self, pattern: MemorablePattern) -> float:
        """Calculate memory confidence from sensor pattern"""
        # Base confidence from pattern
        base_confidence = pattern.confidence
        
        # Weight by sensor reliability
        sensor_weight = self.sensor_weights.get(pattern.sensor_type, 0.5)
        
        # Adjust for pattern significance
        if pattern.is_significant:
            significance_boost = 1.1
        else:
            significance_boost = 0.9
            
        # Final confidence calculation
        memory_confidence = base_confidence * sensor_weight * significance_boost
        
        return min(memory_confidence, 1.0)
        
    def query_sensor_memories(self, 
                            query: str,
                            sensor_types: List[str] = None,
                            time_window: timedelta = None) -> List[Tuple[str, float]]:
        """Query memories derived from sensor data"""
        # Build metadata filter
        metadata_filter = {}
        if sensor_types:
            metadata_filter['sensor_type'] = sensor_types
            
        # Retrieve sensor-derived memories
        memories = self.memory_system.retrieve_with_confidence(
            query=query,
            memory_types=['sensory', 'episodic'],
            limit=20
        )
        
        # Filter by sensor type and time window
        filtered_memories = []
        for memory, weight in memories:
            # Check sensor type
            if sensor_types and memory.metadata.get('sensor_type') not in sensor_types:
                continue
                
            # Check time window
            if time_window:
                age = datetime.now() - memory.timestamp
                if age > time_window:
                    continue
                    
            filtered_memories.append((memory.content, weight))
            
        return filtered_memories
        
    def get_sensor_memory_stats(self) -> Dict:
        """Get statistics about sensor-derived memories"""
        conn = sqlite3.connect(self.memory_system.db_path)
        c = conn.cursor()
        
        # Get counts by sensor type
        c.execute('''
            SELECT 
                json_extract(metadata, '$.sensor_type') as sensor,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM memory_layers
            WHERE json_extract(metadata, '$.sensor_type') IS NOT NULL
            GROUP BY sensor
        ''')
        
        sensor_stats = {}
        for row in c.fetchall():
            sensor, count, avg_conf = row
            sensor_stats[sensor] = {
                'count': count,
                'average_confidence': avg_conf
            }
            
        conn.close()
        
        return {
            'sensor_memory_stats': sensor_stats,
            'total_sensor_memories': sum(s['count'] for s in sensor_stats.values())
        }


# Example usage and testing
if __name__ == "__main__":
    # Create memory system and sensor integration
    memory_system = HierarchicalMemory("sensor_memory_test.db")
    sensor_bridge = SensorMemoryIntegration(memory_system)
    
    # Simulate sensor readings
    print("Testing Sensor Memory Integration...")
    
    # IMU data
    imu_reading = SensorReading(
        sensor_type='imu',
        data={
            'acceleration': [0.1, 0.2, 2.5],  # Sudden movement
            'orientation': {'roll': 10, 'pitch': 5, 'yaw': 90}
        },
        confidence=0.85,
        timestamp=datetime.now()
    )
    
    # Camera data
    camera_reading = SensorReading(
        sensor_type='camera',
        data={
            'detected_objects': [
                {'class': 'person', 'confidence': 0.92, 'attributes': 'wearing blue shirt'},
                {'class': 'laptop', 'confidence': 0.87}
            ],
            'scene_change': 0.7
        },
        confidence=0.90,
        timestamp=datetime.now()
    )
    
    # Audio data
    audio_reading = SensorReading(
        sensor_type='audio',
        data={
            'transcription': 'Hello, how are you today?',
            'transcription_confidence': 0.88,
            'sound_events': [
                {'type': 'door_closing', 'confidence': 0.75, 'is_important': True}
            ]
        },
        confidence=0.80,
        timestamp=datetime.now()
    )
    
    # Process sensor data
    sensor_data = {
        'imu': imu_reading,
        'camera': camera_reading,
        'audio': audio_reading
    }
    
    patterns = sensor_bridge.process_sensor_input(sensor_data)
    
    print(f"\nExtracted {len(patterns)} memorable patterns:")
    for pattern in patterns:
        print(f"  - {pattern.pattern_type}: {pattern.description} (conf: {pattern.confidence:.2f})")
        
    # Query sensor memories
    print("\nQuerying sensor memories...")
    
    # Query for visual memories
    visual_memories = sensor_bridge.query_sensor_memories(
        "person",
        sensor_types=['camera']
    )
    
    print(f"\nVisual memories about 'person':")
    for content, weight in visual_memories:
        print(f"  - {content} (weight: {weight:.2f})")
        
    # Get sensor memory statistics
    stats = sensor_bridge.get_sensor_memory_stats()
    print(f"\nSensor Memory Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Check overall memory health
    health = memory_system.get_memory_health()
    print(f"\nMemory System Health:")
    print(f"  Total memories: {health['total_memories']}")
    print(f"  Average confidence: {health['average_confidence']:.2f}")
    print(f"  Working memory load: {health['working_memory_load']:.1%}")