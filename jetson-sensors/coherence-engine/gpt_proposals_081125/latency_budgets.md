# Latency Budgets & Stability Guardrails

Define end-to-end targets and enforce them with a simple runtime watchdog.

## Targets (suggested defaults)
- Sensor → Fusion: **< 50 ms**
- Fusion → Decision: **< 30 ms**
- Decision → Effector: **< 20 ms**
- End-to-End (S→F→D→E): **< 80 ms** steady-state

Configuration lives in `coherence_engine/config/latency_budgets.yaml`.

## Watchdog Behavior
- Tracks moving averages and p95 for each stage and E2E.
- If any budget is exceeded for `violation_window` consecutive ticks:
  - Flip engine state to **DEGRADED**
  - Trigger **rate/backpressure** policies (see Bridge QoS)
  - Emit a signed log entry (optional) via `ai_collab_hook`
- Auto-recover to **RUNNING** after `recovery_window` consecutive ticks under budget.

## Notes
- Budgets are *operational contracts*; tune per device (e.g., Jetson vs desktop).
- Avoid oscillation: use windows + hysteresis before flipping states.
