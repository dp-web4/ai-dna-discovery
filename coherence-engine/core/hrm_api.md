# HRM API

## Bridge

```
from coherence_engine.adapters.hrm_bridge import HRMBridge, HRMConfig
bridge = HRMBridge(engine, HRMConfig())
bridge.tick(now=time.time())
```

## Telemetry & Witness

- Publishes `hrm/L1` and `hrm/L2` summaries via `engine.telemetry` (if present).
- Emits `engine.witness("hrm_step", {...})` on each step (if present).
