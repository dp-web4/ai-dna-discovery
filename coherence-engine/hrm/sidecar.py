from __future__ import annotations
from typing import Dict

class AffectGate:
    def __init__(self, thresh: float = 0.6, cooldown: float = 1.0, refractory: float = 0.25):
        self.thresh = thresh
        self.cooldown = cooldown
        self.refractory = refractory
        self._last = 0.0

    def should_commit(self, salience: float, now: float) -> bool:
        ok = salience >= self.thresh and (now - self._last) >= (self.cooldown + self.refractory)
        if ok:
            self._last = now
        return ok

class FastWeights:
    def __init__(self):
        self.W: Dict[str, Dict[str, float]] = {}

    def update(self, key: str, vec: Dict[str, float], lr: float = 0.2):
        prev = self.W.get(key, {})
        out: Dict[str, float] = {}
        for k, v in vec.items():
            pv = prev.get(k, 0.0)
            out[k] = pv + lr * (v - pv)
        self.W[key] = out

    def recall(self, key: str) -> Dict[str, float]:
        return dict(self.W.get(key, {}))
