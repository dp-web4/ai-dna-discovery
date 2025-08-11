# Coherence Engine Demo Harness

This demo simulates a simple pipeline on the Jetson-like loop:

```
Sensor → Fusion → Decision → Effector
```

It wires together:
- **LatencyWatchdog** (flip RUNNING↔DEGRADED based on p95 budgets)
- **QoSBridge** (prioritized, TTL-aware message queues with backpressure)

It runs three phases:
1. **Warm / Under Budget** — steady timings, no drops expected.
2. **Overload / Degraded** — injected extra latency + message bursts to trigger DEGRADED.
3. **Recovery** — back under budget; Watchdog should return to RUNNING.

## Run

```bash
python run_demo.py
```

Artifacts:
- Logs: `logs/latency_watchdog.jsonl`
- Console: state flips, enqueue/dequeue behavior, drops due to QoS.
