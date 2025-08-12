# Coherence Dashboard Hooks

No UI dependency; a light collector for the three primary signals:
- `coherence_ema`
- `trust_dispersion`
- `promotions_per_min`

Backends can poll `dashboard.snapshot()` or subscribe to engine telemetry.
