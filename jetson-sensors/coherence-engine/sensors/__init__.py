"""
Sensor implementations for the Coherence Engine.
Both simulated and real hardware sensors available.
"""

# Import real sensors if they exist
try:
    from .real_vision_sensor import RealVisionSensor
    from .real_imu_sensor import RealIMUSensor
    from .persistent_memory_sensor import PersistentMemorySensor
except ImportError:
    pass

# Import stub sensors
from .vision_sensor import VisionSensor
from .imu_sensor import IMUSensor
from .cognition_sensor import CognitionSensor

__all__ = [
    "VisionSensor", "IMUSensor", "CognitionSensor",
    "RealVisionSensor", "RealIMUSensor", "PersistentMemorySensor"
]