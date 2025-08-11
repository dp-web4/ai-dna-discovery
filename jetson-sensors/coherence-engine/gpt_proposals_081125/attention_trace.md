# Attention Trace — Transparency Mode

Enable with `ATTENTION_TRACE=1` (and optional `ATTENTION_TRACE_PATH`).

## Event Schema (JSONL)
```json
{
  "t": 1723335042.123,               // epoch seconds
  "policy": "salience+budget",
  "features_up": {"motion": 0.7, "face": 0.4},
  "features_down": {"background": -0.2},
  "reason": "motion spike; within MRH and under latency budget",
  "weights_pre": {"vision": 0.5, "imu": 0.6},
  "weights_post": {"vision": 0.7, "imu": 0.5},
  "version": 1
}
```
## Privacy
- Do **not** log raw frames or PII; only aggregate features.
- Respect MRH: drop trace events older than the configured horizon when exporting.

## Determinism
- With fixed random seeds and deterministic policies, traces should be reproducible for tests.
