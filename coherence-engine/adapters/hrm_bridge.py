from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from ..hrm.levels import BaseLevel
from ..hrm.sidecar import AffectGate, FastWeights
from ..hrm.training import train_step

@dataclass
class HRMConfig:
    alpha_l1: float = 0.25
    alpha_l2: float = 0.15
    trust_min_l1: float = 0.2
    trust_min_l2: float = 0.3
    lr_l1: float = 0.15
    lr_l2: float = 0.10
    affect_threshold: float = 0.6
    affect_cooldown_sec: float = 1.0
    affect_refractory_sec: float = 0.25

class HRMBridge:
    """Adapter that wires the CE core to HRM L0..L2 levels and sidecar memory."""
    def __init__(self, engine, cfg: HRMConfig | None = None):
        self.engine = engine
        self.cfg = cfg or HRMConfig()
        # Build a minimal HRM bundle on the fly
        class Bundle: pass
        self.hrm = Bundle()
        self.hrm.L0 = BaseLevel("L0", {})
        self.hrm.L1 = BaseLevel("L1", {})
        self.hrm.L2 = BaseLevel("L2", {})
        self.hrm.affect = AffectGate(self.cfg.affect_threshold, self.cfg.affect_cooldown_sec, self.cfg.affect_refractory_sec)
        self.hrm.fast = FastWeights()
        # hyperparams
        self.hrm.alpha_l1 = self.cfg.alpha_l1
        self.hrm.alpha_l2 = self.cfg.alpha_l2
        self.hrm.trust_min_l1 = self.cfg.trust_min_l1
        self.hrm.trust_min_l2 = self.cfg.trust_min_l2
        self.hrm.lr_l1 = self.cfg.lr_l1
        self.hrm.lr_l2 = self.cfg.lr_l2

    def tick(self, now: float):
        """Invoke one HRM training step."""
        return train_step(self.engine, self.hrm, now)
