"""
Lightweight sensor scaffold for the Coherence Engine.
Real sensors can replace these; the interface is a `read(tick=...) -> float` returning [0,1].
"""
from .vision_sensor import VisionSensor
from .imu_sensor import IMUSensor
from .memory_sensor import MemorySensor
from .cognition_sensor import CognitionSensor

__all__ = ["VisionSensor", "IMUSensor", "MemorySensor", "CognitionSensor"]