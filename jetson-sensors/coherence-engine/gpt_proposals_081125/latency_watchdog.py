"""
Latency Watchdog — monitors stage timings and flips engine mode when budgets are violated.
"""
from __future__ import annotations
import time, json, os
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Budgets:
    s2f: int = 50
    f2d: int = 30
    d2e: int = 20
    e2e: int = 80
    violation_window: int = 5
    recovery_window: int = 10
    hysteresis_ms: int = 5

@dataclass
class Stats:
    window: int = 50
    s2f: List[float] = field(default_factory=list)
    f2d: List[float] = field(default_factory=list)
    d2e: List[float] = field(default_factory=list)
    e2e: List[float] = field(default_factory=list)

def p95(xs: List[float]) -> float:
    if not xs: return 0.0
    ys = sorted(xs)
    k = int(0.95 * (len(ys)-1))
    return ys[k]

class LatencyWatchdog:
    def __init__(self, budgets: Budgets | None = None, window: int = 50, log_path: str = "logs/latency_watchdog.jsonl"):
        self.b = budgets or Budgets()
        self.stats = Stats(window=window)
        self.state = "RUNNING"  # or DEGRADED
        self._violations = 0
        self._recovery = 0
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def record(self, s2f_ms: float, f2d_ms: float, d2e_ms: float):
        e2e_ms = s2f_ms + f2d_ms + d2e_ms
        self._push(self.stats.s2f, s2f_ms)
        self._push(self.stats.f2d, f2d_ms)
        self._push(self.stats.d2e, d2e_ms)
        self._push(self.stats.e2e, e2e_ms)
        self._evaluate()

    def _push(self, buf: List[float], val: float):
        buf.append(val)
        if len(buf) > self.stats.window:
            del buf[0]

    def _evaluate(self):
        # compute p95 against budgets + hysteresis
        over = 0
        if p95(self.stats.s2f) > self.b.s2f + self.b.hysteresis_ms: over += 1
        if p95(self.stats.f2d) > self.b.f2d + self.b.hysteresis_ms: over += 1
        if p95(self.stats.d2e) > self.b.d2e + self.b.hysteresis_ms: over += 1
        if p95(self.stats.e2e) > self.b.e2e + self.b.hysteresis_ms: over += 1

        if over:
            self._violations += 1
            self._recovery = 0
        else:
            self._recovery += 1
            self._violations = 0

        if self.state == "RUNNING" and self._violations >= self.b.violation_window:
            self.state = "DEGRADED"
            self._emit_event("DEGRADED")
        elif self.state == "DEGRADED" and self._recovery >= self.b.recovery_window:
            self.state = "RUNNING"
            self._emit_event("RUNNING")

    def _emit_event(self, new_state: str):
        evt = {
            "t": time.time(),
            "state": new_state,
            "p95": {
                "s2f": p95(self.stats.s2f),
                "f2d": p95(self.stats.f2d),
                "d2e": p95(self.stats.d2e),
                "e2e": p95(self.stats.e2e),
            }
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt) + "\n")
