# Phase 1 Integration Patch

This bundle wires:
- **Vectorized trust** via reducer (`weighted_sum`) into `gate_by_trust()` with hysteresis thresholds.
- **SimHash motif IDs** for L1/L2 sidecar commits (keys like `L1:<int64>`).
- **Dashboard metrics** publication from the training loop: `coherence_ema`, `trust_dispersion`, `promotions_per_min`.

Drop-in replacements:
- `coherence_engine/hrm/training.py`
- `coherence_engine/adapters/hrm_bridge.py`

Optional:
- You can replace `weighted_sum` with `logistic` by setting `hrm.trust_reducer` and `hrm.trust_weights` in the bridge.
