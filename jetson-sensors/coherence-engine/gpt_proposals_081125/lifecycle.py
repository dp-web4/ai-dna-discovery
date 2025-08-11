"""
Pattern Lifecycle reference implementation.

Tracks pattern items with support, contradictions, confidence, and recency.
Transitions follow a minimal FSM tuned by policy thresholds.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import time

STATES = ("CANDIDATE", "STABLE", "PROMOTED", "DEPRECATED", "FORGOTTEN")

@dataclass
class Policy:
    support_min: int = 3
    promotion_score: float = 0.75
    contradiction_limit: int = 2
    expiry_sec: int = 300     # seconds of inactivity to decay one level
    mrh_sec: int = 600        # horizon for counting evidence

@dataclass
class PatternItem:
    key: str
    state: str = "CANDIDATE"
    support: int = 0
    contradictions: int = 0
    confidence: float = 0.0
    last_hit: float = field(default_factory=time.time)

class Lifecycle:
    def __init__(self, policy: Optional[Policy] = None):
        self.policy = policy or Policy()
        self.items: Dict[str, PatternItem] = {}

    def observe(self, key: str, consistent: bool, weight: float = 1.0) -> PatternItem:
        it = self.items.get(key) or PatternItem(key=key)
        now = time.time()
        it.last_hit = now
        if consistent:
            it.support += 1
            # logistic-ish update toward 1.0
            it.confidence += (1 - it.confidence) * min(1.0, 0.2 * weight)
        else:
            it.contradictions += 1
            it.confidence *= 0.8  # shrink confidence
        self.items[key] = it
        return self._transition(it, now)

    def tick(self) -> None:
        """Periodic decay and expiry-based downshift."""
        now = time.time()
        for it in list(self.items.values()):
            # passive decay of confidence
            dt = now - it.last_hit
            if dt > self.policy.expiry_sec:
                self._downshift(it)
                # mild decay toward 0.5
                it.confidence += (0.5 - it.confidence) * 0.2
                it.last_hit = now

    def _downshift(self, it: PatternItem):
        order = list(STATES)
        i = order.index(it.state)
        if i < len(order) - 1:
            it.state = order[i + 1]

    def _upshift(self, it: PatternItem):
        order = list(STATES)
        i = order.index(it.state)
        if i > 0:
            it.state = order[i - 1]

    def _transition(self, it: PatternItem, now: float) -> PatternItem:
        p = self.policy
        # promotion logic
        if it.support >= p.support_min and it.confidence >= p.promotion_score:
            if it.state in ("CANDIDATE", "STABLE"):
                it.state = "PROMOTED"
        elif it.support >= p.support_min:
            if it.state == "CANDIDATE":
                it.state = "STABLE"
        # contradiction logic
        if it.contradictions >= p.contradiction_limit and it.state in ("PROMOTED", "STABLE"):
            it.state = "DEPRECATED"
        return it
