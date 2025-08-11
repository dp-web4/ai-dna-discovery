# Trust Calibration — Curves & Tests

This document defines how we calibrate and test **trust weights** per sensor/entity and across contexts.

## Curves
Parameters live in `coherence_engine/config/trust_curves.yaml`:

- **rise**: increase per consistent evidence tick.
- **fall**: decrease per contradiction tick.
- **saturation**: asymptotic cap (0..1).
- **decay_per_sec**: passive drift toward baseline during inactivity.
- **baseline**: starting/idle trust.
- **hysteresis**: buffer to avoid flicker around decision thresholds.
- **adversarial**: optional multipliers and cooldowns when flagged.

## Cross-Context Rule
When two sources agree on a fact within MRH, apply **agree_bonus**; when they conflict, apply **disagree_penalty**.
Weight contributions by recency via an exponential half-life.

## Unit Tests
See `tests/test_trust_curves.py`:

- clean → noisy → adversarial sequences.
- clamp to [0, saturation].
- hysteresis behavior near threshold.
- cross-context bonus/penalty application.

## Dashboard
Expose a sparkline of trust over the last N minutes and show the last event that moved it.
