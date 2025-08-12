from __future__ import annotations
from typing import Dict, Any, Optional
import time
from .residuals import residual_blend
from .metrics import CoherenceMetrics

def error(pred: Dict[str, float], obs: Dict[str, float]) -> Dict[str, float]:
    keys = set(pred) | set(obs)
    return {k: obs.get(k, 0.0) - pred.get(k, 0.0) for k in keys}

def gate_by_trust(vec: Dict[str, float], trust: Dict[str, float], tmin: float = 0.2) -> Dict[str, float]:
    return {k: v for k, v in vec.items() if trust.get(k, 0.0) >= tmin}

def salience(err: Dict[str, float], feat: Dict[str, float], w_err: float = 0.6, w_mag: float = 0.4) -> float:
    if not feat and not err: return 0.0
    mag = sum(abs(v) for v in feat.values()) / max(1, len(feat))
    en = sum(abs(v) for v in err.values()) / max(1, len(err))
    return w_err * en + w_mag * mag

class HRMBundle:
    """Container for levels + sidecar-like pieces; expected attributes: L0,L1,L2, affect, fast, lr_l1, lr_l2, alpha_l1, alpha_l2"""
    pass

def train_step(engine, hrm: HRMBundle, now: Optional[float] = None):
    now = now or time.time()

    # 1) Perception
    l0_in: Dict[str, Any] = getattr(engine, "fused_features")()
    l0_repr = hrm.L0.encode(l0_in)

    # 2) Pattern extraction (L1)
    l1_pred = hrm.L1.predict(None)
    l1_repr = hrm.L1.encode(l0_repr)
    l1_repr = residual_blend(l1_repr, l1_pred, getattr(hrm, "alpha_l1", 0.25))

    # 3) Concept synthesis (L2)
    l2_pred = hrm.L2.predict(None)
    l2_repr = hrm.L2.encode(l1_repr)
    l2_repr = residual_blend(l2_repr, l2_pred, getattr(hrm, "alpha_l2", 0.15))

    # 4) Errors
    e1 = error(l1_pred, l1_repr)
    e2 = error(l2_pred, l2_repr)

    # 5) Update
    hrm.L1.update(e1)
    hrm.L2.update(e2)

    # 6) Trust gating (placeholder trust maps)
    t1 = getattr(hrm.L1.state, "trust", {})
    t2 = getattr(hrm.L2.state, "trust", {})
    gated_l1 = gate_by_trust(l1_repr, t1, getattr(hrm, "trust_min_l1", 0.2))
    gated_l2 = gate_by_trust(l2_repr, t2, getattr(hrm, "trust_min_l2", 0.3))

    # 7) Sidecar commits
    s1 = salience(e1, gated_l1); s2 = salience(e2, gated_l2)
    if hrm.affect.should_commit(s1, now): hrm.fast.update(f"L1:{len(gated_l1)}", gated_l1, lr=getattr(hrm, "lr_l1", 0.15))
    if hrm.affect.should_commit(s2, now): hrm.fast.update(f"L2:{len(gated_l2)}", gated_l2, lr=getattr(hrm, "lr_l2", 0.10))

    # 8) Telemetry / witness hooks (optional)
    if hasattr(engine, "telemetry"):
        engine.telemetry.publish("hrm/L1", {"k": len(gated_l1)})
        engine.telemetry.publish("hrm/L2", {"k": len(gated_l2)})
    if hasattr(engine, "witness"):
        engine.witness("hrm_step", {"e1": sum(abs(v) for v in e1.values()), "e2": sum(abs(v) for v in e2.values())})

    return {"L1": gated_l1, "L2": gated_l2, "errors": {"e1": e1, "e2": e2}}
