# HRM × Coherence Engine — Condensed Implementation Plan (L0–L2 First)

**Date:** 2025-08-12  
**Scope:** Integrate HRM-style hierarchical memory with the Coherence Engine (CE) using a sidecar pattern and staged rollout (Levels 0–2 first). Ready for Jetson tests.

---

## 0) Guiding Principles
- **Start shallow (L0–L2)**: stabilize perception→pattern→concept loop before L3/L4.
- **Sidecar, not surgery**: HRM augments CE via modular adapters; CE API remains consistent.
- **Trust = attention**: use dynamic trust weights to gate signals between layers.
- **Witness everything**: emit lightweight witness marks on key transitions for provenance.
- **Dampen feedback**: avoid oscillations with hysteresis, EMA filters, and bounded learning rates.

---

## 1) Repo Layout Additions (proposed)
```
coherence_engine/
  hrm/
    __init__.py
    levels.py            # L0/L1/L2 abstractions
    residuals.py         # cross-level skip connections
    sidecar.py           # fast weights + affect gating
    training.py          # train_step(), error backprop stubs
    metrics.py           # coherence metrics per level
    config/
      hrm_levels.yaml    # enables levels, params per level
      hrm_learning.yaml  # lr, decay, thresholds
  adapters/
    hrm_bridge.py        # glue between CE core and HRM sidecar
docs/
  hrm_overview.md
  hrm_api.md
tests/
  test_hrm_levels.py
  test_hrm_sidecar.py
  test_hrm_training_loop.py
```

---

## 2) Level Semantics (L0–L2)
- **L0 (Sensor Field)**: fused, denoised features from sensors (vision/IMU/temp…). Own trust curve.  
- **L1 (Pattern Field)**: short-lived motifs (edges, motion vectors, periodicities, simple correlations).  
- **L2 (Concept Field)**: stable aggregates (object IDs, zones, conditions like “charging”, “overheat risk”).

> Each level has: `state`, `trust`, `encode()`, `predict()`, `update(error)`, `summarize()`.

---

## 3) Data Structures (pydantic-ish sketches)

```python
# coherence_engine/hrm/levels.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class LevelState:
    name: str
    features: Dict[str, float]      # current representation
    trust: Dict[str, float]         # per-feature/source trust
    timestamp: float
    meta: Dict[str, Any]

class BaseLevel:
    def __init__(self, name:str, cfg:dict):
        self.name = name
        self.cfg = cfg
        self.state = LevelState(name, {}, {}, 0.0, {})

    def encode(self, inputs:Dict[str, Any]) -> Dict[str,float]: ...
    def predict(self, higher_ctx:Optional[Dict]=None) -> Dict[str,float]: ...
    def update(self, error:Dict[str,float]) -> None: ...
    def summarize(self) -> Dict[str, Any]: return {"name": self.name, "k": len(self.state.features)}
```

---

## 4) Residual Connections (skip links)
```python
# coherence_engine/hrm/residuals.py
def residual_blend(lower_repr, higher_hint, alpha:float=0.25):
    # convex blend with clamp
    out = {}
    for k,v in lower_repr.items():
        h = higher_hint.get(k, v)
        out[k] = (1-alpha)*v + alpha*h
    return out
```

Tunable `alpha` per level in `hrm_levels.yaml` (see §8).

---

## 5) Sidecar Memory (Fast Weights + Affect Gating)
```python
# coherence_engine/hrm/sidecar.py
class AffectGate:
    def __init__(self, thresh=0.6, cooldown=1.0):
        self.thresh, self.cooldown, self.last = thresh, cooldown, 0.0
    def should_commit(self, salience, now):
        ok = salience >= self.thresh and (now - self.last) >= self.cooldown
        if ok: self.last = now
        return ok

class FastWeights:
    def __init__(self): self.W = {}  # key->vector
    def update(self, key, vec, lr=0.2):
        prev = self.W.get(key, {})
        self.W[key] = {k: prev.get(k,0.0) + lr*(v - prev.get(k,0.0)) for k,v in vec.items()}
    def recall(self, key): return self.W.get(key, {})
```

- Hook: L1 commits motifs keyed by hashed pattern IDs; L2 commits concept vectors keyed by concept ID.
- Salience = function of surprise + trust + task relevance.

---

## 6) Training Loop (pseudocode)
```python
# coherence_engine/hrm/training.py
def train_step(engine, hrm, now):
    # 1) Perception (CE core already does fusion → provide L0 input features)
    l0_in = engine.fused_features()                   # from CE sensors
    l0_repr = hrm.L0.encode(l0_in)

    # 2) Pattern extraction
    l1_pred = hrm.L1.predict(higher_ctx=None)         # prior
    l1_repr = hrm.L1.encode(l0_repr)                  # encode from L0
    l1_repr = residual_blend(l1_repr, l1_pred, hrm.alpha_l1)

    # 3) Concept synthesis
    l2_pred = hrm.L2.predict(higher_ctx=None)
    l2_repr = hrm.L2.encode(l1_repr)
    l2_repr = residual_blend(l2_repr, l2_pred, hrm.alpha_l2)

    # 4) Compute prediction errors
    e1 = error(l1_pred, l1_repr)                      # L1 error
    e2 = error(l2_pred, l2_repr)                      # L2 error

    # 5) Update (local learning)
    hrm.L1.update(e1); hrm.L2.update(e2)

    # 6) Trust as attention (gate upward flow)
    gated_l1 = gate_by_trust(l1_repr, hrm.L1.state.trust)
    gated_l2 = gate_by_trust(l2_repr, hrm.L2.state.trust)

    # 7) Sidecar commits (affect-gated)
    s1 = salience(e1, gated_l1); s2 = salience(e2, gated_l2)
    if hrm.affect.should_commit(s1, now): hrm.fast.update(keyL1(gated_l1), gated_l1, lr=hrm.lr_l1)
    if hrm.affect.should_commit(s2, now): hrm.fast.update(keyL2(gated_l2), gated_l2, lr=hrm.lr_l2)

    # 8) Emit summaries + witness marks
    engine.telemetry.publish("hrm/L1", hrm.L1.summarize())
    engine.telemetry.publish("hrm/L2", hrm.L2.summarize())
    engine.witness("hrm_step", {"e1":norm(e1), "e2":norm(e2)})
```

Helpers:
```python
def error(pred, obs): return {k: obs.get(k,0)-pred.get(k,0) for k in set(pred)|set(obs)}
def gate_by_trust(vec, trust, tmin=0.2): return {k:v for k,v in vec.items() if trust.get(k,0)>=tmin}
def norm(e): return sum(abs(v) for v in e.values())/max(1,len(e))
def salience(err, feat, w_err=0.6, w_mag=0.4):
    return w_err*norm(err) + w_mag*(sum(abs(v) for v in feat.values())/max(1,len(feat)))
```

---

## 7) CE Adapter (glue)
```python
# coherence_engine/adapters/hrm_bridge.py
class HRMBridge:
    def __init__(self, engine, cfg_levels, cfg_learn):
        self.engine = engine
        self.hrm = build_levels(cfg_levels, cfg_learn)  # constructs L0..L2, sidecar

    def tick(self, now):
        # integrate with watchdog + QoS (optional)
        # timings measured around train_step for S→F→D→E
        return train_step(self.engine, self.hrm, now)
```

Register this bridge as a CE plugin so it starts after sensors/vision are RUNNING.

---

## 8) Config (YAML stubs)

**`hrm_levels.yaml`**
```yaml
levels:
  L0:
    enabled: true
  L1:
    enabled: true
    alpha_residual: 0.25
    trust_min: 0.2
  L2:
    enabled: true
    alpha_residual: 0.15
    trust_min: 0.3
```

**`hrm_learning.yaml`**
```yaml
learning:
  lr_l1: 0.15
  lr_l2: 0.10
  affect_threshold: 0.6
  affect_cooldown_sec: 1.0
  max_grad_norm: 1.0   # (for future param models)
```

---

## 9) Coherence Metrics (per level)
Expose via telemetry:  
- **Coherence score**: 1 - normalized prediction error (EMA over window).  
- **Trust dispersion**: variance of trust weights (prefer neither collapse nor uniform).  
- **Promotion rate** (L1→L2): motifs promoted per minute.  
- **Forgetting rate**: deprecated patterns per minute.  
- **Stability index**: fraction of ticks in STABLE vs MOVING/UNSTABLE.

```python
# coherence_engine/hrm/metrics.py
class CoherenceMetrics:
    def __init__(self): self.ema = 0.0
    def update(self, err, beta=0.9):
        val = 1.0 - min(1.0, norm(err))  # 0..1
        self.ema = beta*self.ema + (1-beta)*val
        return self.ema
```

---

## 10) Stability Controls
- **Hysteresis** on state changes (RUNNING↔DEGRADED).  
- **EMA filters** on trust and coherence.  
- **Learning rate clamps** and **gradient clipping** (for learned submodules).  
- **Cooldown** after large errors before acting on them.  
- **Backpressure** via QoS + latency watchdog (already added).

---

## 11) Evaluation & Tests
- **Unit**: `test_hrm_levels.py` (encode/predict/update invariants), `test_hrm_sidecar.py` (affect gating), `test_hrm_training_loop.py` (error decreases under synthetic patterns).  
- **Integration**: Jetson loop with replayed sensor logs; verify improvement in coherence EMA and promotion of true motifs.  
- **Ablations**: disable residuals; disable sidecar; randomize trust → compare metrics.

---

## 12) Milestones
1. **M0 (Day 1–2)**: Skeleton modules + configs + adapter; compile and run `tick()` no-op.  
2. **M1 (Day 3–4)**: L1 motifs from L0, sidecar commit/recall; telemetry flowing.  
3. **M2 (Day 5–6)**: L2 concepts; residuals tuned; coherence metric improving on replay.  
4. **M3 (Week 2)**: On-device Jetson tests; watchdog integration; witness marks for key events.  
5. **M4 (Week 3)**: Add L3 narrative and L4 intention (gated until L0–L2 stable).

---

## 13) Wiring to Existing Components
- Use **trust_curves.yaml** for `trust_min` gating at each level.  
- Emit **attention_trace** events when residual blending or gating materially changes weights.  
- Route **pattern_lifecycle** promotions from L1 to L2 (already scaffolded).  
- Record **conflict resolution** outcomes as L2 concept confirmations/contradictions.

---

## 14) Witness & Provenance
On each `train_step` where `salience >= threshold` and a commit occurs:  
- Append entry to `ai_collab_log.md` (HMAC or ed25519).  
- Emit **witness mark**: `{level, key, ts, err_norm, trust_snapshot_hash}`.  
- Optionally aggregate to the **Memory Lightchain** (leaf → parent).

---

## 15) Open Questions
- Best hash/keying for motifs (L1) to balance collision vs. memory.  
- When should top-down predictions be generated (every tick vs adaptive)?  
- How to expose L2 concepts to effectors safely (latency budgets)?  
- Where to keep long-term L4 without bloat (compression / pruning policies).

---

*— Implementation-first, spec-light: ship L0–L2 loop, measure, then grow.*
