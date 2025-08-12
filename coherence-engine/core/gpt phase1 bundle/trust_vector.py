"""
Vectorized Trust (T3/V3) with scalar reducers.

Each source can provide a trust vector keyed by traits/topics/roles.
Reducers map vectors → scalar weights for gating/attention.

Example trust vector:
{
  "reliability": 0.8,
  "recency": 0.7,
  "expertise.motion": 0.9,
  "expertise.temperature": 0.4
}
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable
import math

@dataclass
class TrustVector:
    values: Dict[str, float]

    def clamp(self) -> 'TrustVector':
        self.values = {k: max(0.0, min(1.0, float(v))) for k, v in self.values.items()}
        return self

    def keys(self) -> Iterable[str]:
        return self.values.keys()

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.values.get(key, default))

# Common reducers
def weighted_sum(tv: TrustVector, weights: Dict[str, float]) -> float:
    s = 0.0
    w = 0.0
    for k, v in tv.values.items():
        wk = float(weights.get(k, 0.0))
        s += v * wk
        w += abs(wk)
    return max(0.0, min(1.0, s / w)) if w > 0 else 0.0

def geometric_mean(tv: TrustVector, keys: Iterable[str]) -> float:
    vals = [max(1e-6, min(1.0, tv.get(k))) for k in keys]
    if not vals: return 0.0
    g = 1.0
    for v in vals: g *= v
    return max(0.0, min(1.0, g ** (1.0 / len(vals))))

def min_gate(tv: TrustVector, keys: Iterable[str]) -> float:
    vals = [tv.get(k, 0.0) for k in keys]
    return min(vals) if vals else 0.0

def logistic(tv: TrustVector, weights: Dict[str, float], bias: float = 0.0) -> float:
    # σ(w·x + b) using simple dot over selected keys (weights define selection)
    z = bias
    for k, w in weights.items():
        z += w * tv.get(k, 0.0)
    return 1.0 / (1.0 + math.exp(-z))

__all__ = ["TrustVector", "weighted_sum", "geometric_mean", "min_gate", "logistic"]
