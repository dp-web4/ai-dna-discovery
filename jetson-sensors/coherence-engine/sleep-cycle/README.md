# HRM Sleep Cycle Skeleton

This project is a minimal runnable scaffold that simulates the pipeline for:
1. Capturing multi-modal experience data from a runtime (vision, IMU, coherence).
2. Logging that data into rolling storage.
3. "Sleeping" to harvest experience into context→target training samples.
4. Training a toy model (GRU-based) as a stand-in for a full HRM block.

## Quickstart

```bash
pip install torch pandas pyarrow
python src/runtime.py
python src/sleep_cycle.py
```

- `runtime.py` simulates a coherence engine emitting packets.
- `sleep_cycle.py` collects the log, generates samples, and trains the toy model.

Replace the dummy event generator in `runtime.py` with your Jetson coherence engine,
and swap the toy GRU in `sleep_cycle.py` for the HRM architecture.
