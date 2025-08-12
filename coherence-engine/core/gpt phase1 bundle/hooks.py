"""
Coherence dashboard hooks (no UI dependency).
Collects rolling metrics so any UI/telemetry backend can render them.
"""
from __future__ import annotations
from collections import deque
from typing import Deque, Dict, Any, Tuple

class RollingMetric:
    def __init__(self, maxlen: int = 300):
        self.values: Deque[Tuple[float, float]] = deque(maxlen=maxlen)  # (t, v)

    def add(self, t: float, v: float):
        self.values.append((t, float(v)))

    def last(self) -> float:
        return self.values[-1][1] if self.values else 0.0

class CoherenceDashboard:
    def __init__(self):
        self.streams: Dict[str, RollingMetric] = {
            "coherence_ema": RollingMetric(),
            "trust_dispersion": RollingMetric(),
            "promotions_per_min": RollingMetric(),
        }

    def record(self, name: str, t: float, v: float):
        if name not in self.streams:
            self.streams[name] = RollingMetric()
        self.streams[name].add(t, v)

    def snapshot(self) -> Dict[str, float]:
        return {k: m.last() for k, m in self.streams.items()}
