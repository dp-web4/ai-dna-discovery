# Bridge QoS & Backpressure

Message classification and flow control across the engine's internal and external bridges.

## Message Classes
- `state` — periodic summaries; low priority, drop-oldest allowed.
- `event` — important changes; medium priority, best-effort deliver.
- `experience_batch` — bulk data (e.g., embeddings); lowest priority, compressible and droppable.
- `pattern_diff` — model updates; high priority, must-deliver within TTL.

## Policies
Each class defines:
- `priority` (0..3)
- `ttl_ms` — time-to-live
- `drop_policy` — `drop_oldest`, `drop_new`, or `block`
- `queue_max` — bounded queue length

Defaults live in `coherence_engine/config/bridge_qos.yaml`.

## Behavior
- Enqueue applies class policy immediately.
- A housekeeper periodically expires messages past TTL.
- Backpressure escalates:
  1. Drop according to class policy
  2. Signal upstream to slow rate (if supported)
  3. Notify Watchdog to reduce load or enter DEGRADED

See `coherence_engine/bridge/qos.py` for reference implementation.
