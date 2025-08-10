#!/usr/bin/env python3
"""
Base Sensor Interface for Coherence Engine
All sensors (spatial and temporal) inherit from this
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import time

@dataclass
class SensorReading:
    """Standard format for all sensor readings"""
    sensor_type: str           # vision, imu, memory, cognition, etc
    timestamp: float           # Unix timestamp
    data: Dict[str, Any]      # Sensor-specific data
    confidence: float         # 0.0 to 1.0
    relevance: float         # Context-based relevance 0.0 to 1.0
    metadata: Dict[str, Any]  # Additional sensor metadata

class BaseSensor(ABC):
    """Abstract base class for all sensors"""
    
    def __init__(self, sensor_type: str, name: str):
        self.sensor_type = sensor_type
        self.name = name
        self.is_active = False
        self.trust_score = 0.5  # Start with neutral trust
        self.trust_history = []
        self.last_reading = None
        self.error_count = 0
        self.success_count = 0
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the sensor hardware/connection"""
        pass
    
    @abstractmethod
    def read(self) -> Optional[SensorReading]:
        """Get current sensor reading"""
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate the sensor if needed"""
        pass
    
    def update_trust(self, success: bool, weight: float = 0.1):
        """Update trust score based on sensor performance"""
        if success:
            self.success_count += 1
            self.trust_score = min(1.0, self.trust_score + weight)
        else:
            self.error_count += 1
            self.trust_score = max(0.0, self.trust_score - weight)
        
        self.trust_history.append({
            'timestamp': time.time(),
            'trust': self.trust_score,
            'success': success
        })
        
        # Keep only last 100 trust updates
        if len(self.trust_history) > 100:
            self.trust_history.pop(0)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get sensor health and statistics"""
        total_attempts = self.success_count + self.error_count
        success_rate = self.success_count / total_attempts if total_attempts > 0 else 0
        
        return {
            'sensor_type': self.sensor_type,
            'name': self.name,
            'is_active': self.is_active,
            'trust_score': self.trust_score,
            'success_rate': success_rate,
            'error_count': self.error_count,
            'success_count': self.success_count,
            'last_reading_time': self.last_reading.timestamp if self.last_reading else None
        }
    
    def shutdown(self):
        """Clean shutdown of sensor"""
        self.is_active = False

class SpatialSensor(BaseSensor):
    """Base class for spatial sensors (vision, imu, audio)"""
    
    def __init__(self, sensor_type: str, name: str):
        super().__init__(sensor_type, name)
        self.sampling_rate = 30  # Hz
        self.spatial_resolution = None
        
    @abstractmethod
    def get_spatial_data(self) -> Dict[str, Any]:
        """Get spatial-specific data"""
        pass

class TemporalSensor(BaseSensor):
    """Base class for temporal sensors (memory, cognition)"""
    
    def __init__(self, sensor_type: str, name: str):
        super().__init__(sensor_type, name)
        self.time_window = 100  # How many timesteps to consider
        self.temporal_buffer = []
        
    @abstractmethod
    def get_temporal_context(self) -> Dict[str, Any]:
        """Get temporal-specific context"""
        pass
    
    @abstractmethod
    def predict_future(self, timesteps: int) -> Dict[str, Any]:
        """Predict future states"""
        pass
    
    def add_to_buffer(self, data: Any):
        """Add data to temporal buffer"""
        self.temporal_buffer.append({
            'timestamp': time.time(),
            'data': data
        })
        
        # Keep only time_window entries
        if len(self.temporal_buffer) > self.time_window:
            self.temporal_buffer.pop(0)