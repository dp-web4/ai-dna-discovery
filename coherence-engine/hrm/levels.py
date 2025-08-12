from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time

@dataclass
class LevelState:
    name: str
    features: Dict[str, float]
    trust: Dict[str, float]   # could be vectorized later; keep scalar per-feature for now
    timestamp: float
    meta: Dict[str, Any]

class BaseLevel:
    def __init__(self, name: str, cfg: dict | None = None):
        self.name = name
        self.cfg = cfg or {}
        self.state = LevelState(name, {}, {}, 0.0, {})

    def encode(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Map raw inputs to a compact representation for this level.
        Override in subclasses. Here we pass through numeric items.
        """
        out: Dict[str, float] = {}
        for k, v in inputs.items():
            if isinstance(v, (int, float)):
                out[k] = float(v)
        self.state.features = out
        self.state.timestamp = time.time()
        return out

    def predict(self, higher_ctx: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Produce a prior for the next tick. Default prior is last features.
        """
        return dict(self.state.features)

    def update(self, error: Dict[str, float]) -> None:
        """
        Update internal parameters from prediction error. Stub: apply small correction.
        Replace with learned parameters when available.
        """
        kappa = float(self.cfg.get("learn_rate", 0.05))
        for k, e in error.items():
            self.state.features[k] = self.state.features.get(k, 0.0) + kappa * e

    def summarize(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "k": len(self.state.features),
            "t": self.state.timestamp,
        }
