"""
Coherence Engine Plugin System
August 11, 2025
"""

from .base import PluginBase, SensorBase, EffectorBase, SensorEffectorBridge
from .plugin_manager import PluginManager

__all__ = [
    'PluginBase',
    'SensorBase', 
    'EffectorBase',
    'SensorEffectorBridge',
    'PluginManager'
]

__version__ = '1.0.0'