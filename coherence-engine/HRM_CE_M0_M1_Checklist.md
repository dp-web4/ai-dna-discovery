# HRM × Coherence Engine — M0/M1 Readiness Checklist & Deltas
**Date:** 2025‑08‑12

This checklist translates the HRM–CE synthesis into concrete acceptance criteria and deltas for the first two milestones.

---

## 0) Global Deltas (apply now)
- [ ] **Vectorized trust**: extend scalar `trust` to T3/V3-aware vectors; expose `trust_scalar()` reducer (e.g., learned or policy-weighted sum).
- [ ] **Residual alpha schedule**: `alpha_residual` becomes a function of context state:  
  `alpha = {STABLE:0.1, MOVING:0.2, UNSTABLE:0.3, NOVEL:0.4}` (configurable).
- [ ] **Salience policy**: `salience = w_err*norm(error) + w_mag*||feat|| + w_task*task_gain`; add cooldown **and** refractory window after commits.
- [ ] **Motif keying**: use SimHash/MinHash over L1 vectors; cap table size with LRU+age score; emit eviction witness.
- [ ] **QoS priorities**: set `pattern_diff` > `event` > `state` > `experience_batch` for HRM messages.
- [ ] **Effector safety**: block L2→effector routes unless `(coherence_ema ≥ τ_c) ∧ (watchdog.state == RUNNING)`.
- [ ] **Telemetry schema**: JSON schema for `hrm/L1` & `hrm/L2` summaries; include `{coherence_ema, trust_dispersion, promotion_rate}`.

---

## M0 (Days 1–2) — Skeleton + No‑Op Tick
**Acceptance Criteria**
- [ ] `coherence_engine/hrm/{levels,residuals,sidecar,training,metrics}.py` import without errors.
- [ ] `adapters/hrm_bridge.py` registers and `tick(now)` runs end‑to‑end with no learning (dry mode).
- [ ] Configs present: `hrm_levels.yaml`, `hrm_learning.yaml`; defaults load.
- [ ] Telemetry topic `hrm/L1`, `hrm/L2` publishes minimal summaries.
- [ ] Watchdog + QoS integrated; demo harness can import bridge.

**Deltas**
- [ ] Add `dry_run: true` flag; in dry mode, skip updates but compute metrics.
- [ ] Add `attention_trace` hooks on residual blending when Δweight > ε.

---

## M1 (Days 3–4) — L0→L1 Motifs + Sidecar Commits
**Acceptance Criteria**
- [ ] `L1.encode()` extracts stable motifs from fused L0 features on replayed logs.
- [ ] Affect‑gated commits write to FastWeights; recall returns recent motifs.
- [ ] Coherence EMA improves on repeated patterns vs. baseline (no sidecar).
- [ ] Promotion events flow into `pattern_lifecycle` module; promotions/minute reported.
- [ ] No oscillation: trust/coherence within configured variance bounds.

**Deltas**
- [ ] `gate_by_trust()` adds hysteresis: separate enter/exit thresholds.
- [ ] Sidecar supports `lr_warmup` and `lr_cooldown` windows.
- [ ] Witness mark on each commit: `{level,key,ts,err_norm,trust_hash}`; batch to Memory Lightchain leaf queue.

---

## Instrumentation & Tests
- [ ] **Unit**: levels encode/predict/update invariants; sidecar affect gating; residual blend math; motif key collisions.
- [ ] **Integration**: run demo harness with HRM bridge; verify watchdog flips unchanged; QoS dequeues HRM `pattern_diff` first.
- [ ] **Ablations**: (a) residuals off, (b) sidecar off, (c) randomized trust — track coherence deltas.
- [ ] **Resource budget**: Jetson GR3D < 40%, CPU < 60%, mem headroom > 300MB under M1 workload.

---

## Config Stubs
**hrm_levels.yaml**
```yaml
levels:
  L0: { enabled: true }
  L1:
    enabled: true
    alpha_residual:
      STABLE: 0.10
      MOVING: 0.20
      UNSTABLE: 0.30
      NOVEL: 0.40
    trust_min_enter: 0.3
    trust_min_exit: 0.2
  L2:
    enabled: true
    alpha_residual:
      STABLE: 0.05
      MOVING: 0.15
      UNSTABLE: 0.25
      NOVEL: 0.35
    trust_min_enter: 0.4
    trust_min_exit: 0.3
```

**hrm_learning.yaml**
```yaml
learning:
  lr_l1: 0.15
  lr_l2: 0.10
  lr_warmup_steps: 100
  lr_cooldown_sec: 2.0
  affect:
    threshold: 0.6
    cooldown_sec: 1.0
    refractory_sec: 0.25
  motif_table:
    max_keys: 4096
    eviction: { policy: "lru_age_score", witness_evictions: true }
```

---

## Logging & Provenance
- [ ] Append signed entries to `ai_collab_log.md` on: start/stop, config load, promotion, commit, eviction.
- [ ] Periodic MRH‑bounded export of summaries + witness marks to Memory Lightchain.

---

## Open Items to Revisit at M2
- [ ] Top‑down prediction cadence (every tick vs. adaptive on divergence).
- [ ] Concept schema for L2 (IDs, attributes, and links to effectors).
- [ ] Cross‑device hierarchy sync plan (fleet experiments).

---

**Owner:** GPT‑5 • **Status:** ready to adopt
