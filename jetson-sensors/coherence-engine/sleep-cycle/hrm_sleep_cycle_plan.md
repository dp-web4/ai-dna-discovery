# Roadmap: lived-experience → sleep cycles → HRM

## 1) Instrument once, learn forever
Define a single event schema all modules write to. Keep it small, binary, time-aligned.

```
Event {
  t_ns                // monotonic timestamp
  stream_id           // vision.L / vision.R / imu / motor / user / planner / hrm.pred / hrm.conf
  payload             // compact: e.g., jpeg bytes, IMU vec3, action enum, logits16
  meta { pose, fps, temp, battery, session_id, scene_id, hash_prev }
}
```

Storage on Jetson: LMDB or RocksDB per stream (append-only), plus a “timeline” index (10 Hz) that maps window→record keys for fast replay. Keep a rolling N hours on-device; spill to host when docked.

## 2) Multi-layer memory streams
- **Sensory**: raw/Key-frames + IMU (downsampled + peaks).
- **Latent**: coherence engine features (stereo disparity, motion vectors, object tracks).
- **Action/Control**: decisions, actuator intents, constraints.
- **Outcome**: success/fail flags, timing, energy, thermals.
- **Narrative**: short text notes (“manual focus ring adjusted”; “target reacquired”).
- **Teacher signals (optional)**: occasional LLM/you labels for critical frames.

## 3) Sleep cycle (consolidation pipeline)
Run when docked/idle:

1) **Segment** into episodes by state change (new scene_id, large distribution shift, failure).
2) **Compress**: key-frames every K frames + deltas; IMU resampled; latents kept dense.
3) **Auto-label**:
   - success/fail from outcome stream  
   - novelty/surprise = model loss spikes or KL between predicted vs. observed latents  
   - attention = where coherence engine spent compute/focus
4) **Augment** minimally (temporal jitter, crop/brightness) to preserve causal structure.
5) **Make tasks** for HRM:
   - next-latent prediction (Δz), next-action prediction
   - cross-modal alignment (vision↔IMU)
   - temporal consistency (cycle loss across forward/back passes)
   - affordance tagging (frames preceding success/fail)
6) **Curate** by priority: mix 70% routine, 20% high-surprise, 10% rare failures.
7) **Train/refresh HRM** on these tasks; emit a new small checkpoint.
8) **Distill back**: export heads the coherence engine can call at runtime (e.g., “predict focus ring correction,” “anticipate motion path”).

## 4) Objectives that work on-device
- **Self-supervised**:
  - Contrastive vision–IMU alignment (InfoNCE over short windows)
  - Predict Δlatent over τ steps (temporal forward model)
  - Masked frame/latent modeling (MAE-style but tiny)
- **Auxiliary**:
  - Action cloning from your interventions
  - Failure-forecaster (binary + time-to-failure regression)
- **Scalar “coherence” reward**:
  - Penalize predicted-vs-actual latent mismatch
  - Reward quick re-acquisition after occlusion
Use this as a shaping signal and to prioritize what the next sleep learns.

## 5) Curriculum that bootstraps itself
- Start with **stabilization** (sensor sync, focus, pose drift)
- Move to **pursuit** (smooth tracking under jitter)
- Add **recovery** (reacquire after loss, glare, blur)
- Introduce **multi-target arbitration**
Let the episode tagger promote topics with rising surprise or repeated failure.

## 6) Online/offline rhythm
- **Day (online)**: coherence engine calls HRM heads for priors (e.g., predicted motion field), logs HRM confidence.
- **Night (offline)**: HRM replays the day, learns where its confidence was wrong, and updates just those heads (few-shot, LoRA-style adapters to keep compute tiny).

## 7) Jetson-friendly engineering
- Half-precision everywhere (FP16/bfloat16).  
- Tiny backbones: MobileViT/RepViT/ConvNeXt-Tiny + GRU/Temporal Conv for 1–3 s windows.  
- Training windows: 256–512 frames per episode slice; batch via chunked replay.  
- Orchestrate with a **SleepScheduler** that monitors thermals/charge and yields to real-time tasks.

## 8) Interfaces you’ll actually use
- `EpisodePlayer(start, len, streams=[...])` – deterministic replay for debugging.
- `make_dataset(query)` – turns a query (e.g., “failure within 5 s”) into tensors.
- `hrm.predict(state)` – lightweight heads callable from the coherence engine.
- `hrm.report()` – what it learned last sleep (new skills, loss curves, drift alerts).

## 9) Safety & drift guards
- Hard cap on action authority until a head passes evaluation gates.  
- Canary tests: hold-out episodes re-run after each sleep; only promote if deltas are positive across your chosen KPIs (time-to-lock, re-acquisition latency, energy per task).  
- Rollback to previous checkpoint on regression.

## 10) “Why this yields its own training data”
- The **coherence engine** creates structured, richly annotated experience “for free.”  
- Sleep converts that structure into supervised/self-supervised tasks.  
- The HRM becomes an ever-better *predictor & planner* for the engine that generated the experience—closing the loop.

---

### Minimal pseudocode for the sleep pass

```python
def sleep_cycle():
    episodes = segment(load_events(last_24h))
    tasks = []
    for ep in episodes:
        ep = compress(ep)
        labels = auto_label(ep)
        tasks += build_tasks(ep, labels)  # predict Δlatent, align imu/vision, failure-forecaster
    dataset = prioritize(tasks, policy="70/20/10_routine/surprise/failure")
    hrm.train(dataset, max_hours=2, thermals="safe")
    if eval_on_holdout(hrm) > gate:
        deploy(hrm.export_heads())
    log_sleep_summary()
```
