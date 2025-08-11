"""
Weighted conflict resolution across entities.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math, time

@dataclass
class Source:
    id: str
    trust: float         # 0..1 (dynamic)
    expertise: float     # 0..1 (static or learned)
    role_priority: int = 0
    t: float = None      # timestamp of observation

def recency_weight(t_obs: float, half_life: float) -> float:
    dt = max(0.0, time.time() - t_obs)
    return math.exp(-dt / half_life) if half_life > 0 else 1.0

def resolve(hypotheses: Dict[str, List[Source]], quorum: float = 1.2, half_life: float = 120.0) -> Tuple[str, Dict[str, float]]:
    scores: Dict[str, float] = {}
    total = 0.0
    for hyp, sources in hypotheses.items():
        s = 0.0
        for src in sources:
            rw = recency_weight(src.t or time.time(), half_life)
            s += max(0.0, src.trust) * max(0.0, src.expertise) * rw
        scores[hyp] = s
        total += s

    # quorum check
    if total < quorum:
        return ("inconclusive", scores)

    # pick max; tie-break by role_priority sum
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    # detect ties within small epsilon
    eps = 1e-9
    tied = [h for h,v in scores.items() if abs(v - scores[best]) < eps]
    if len(tied) > 1:
        # role priority sum
        rp = {h: sum(src.role_priority for src in hypotheses[h]) for h in tied}
        best = max(rp.items(), key=lambda kv: kv[1])[0]
    return (best, scores)
