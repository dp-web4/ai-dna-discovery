"""
Hierarchical Residual Memory (HRM) module for Coherence Engine.

This module implements hierarchical consciousness layers with:
- L0: Raw sensor fusion
- L1: Pattern extraction
- L2: Concept synthesis
- L3: Narrative construction (future)
- L4: Intention formation (future)
"""

from .levels import BaseLevel, LevelState
from .sidecar import AffectGate, FastWeights
from .training import train_step, error, gate_by_trust, salience
from .residuals import residual_blend
from .metrics import CoherenceMetrics, norm

__all__ = [
    'BaseLevel',
    'LevelState',
    'AffectGate',
    'FastWeights',
    'train_step',
    'error',
    'gate_by_trust',
    'salience',
    'residual_blend',
    'CoherenceMetrics',
    'norm',
]

__version__ = '0.1.0'