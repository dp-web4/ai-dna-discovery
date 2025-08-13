# Trust Dynamics Experiment
*August 12, 2025*

## Hypothesis
Trust weights in the coherence engine adapt dynamically when sensors provide conflicting information. The system should learn to reduce trust in unreliable sensors and increase trust in consistent ones.

## Test Design

### Phase 1: Baseline (30 seconds)
- All sensors operating normally
- Establish baseline trust weights
- Record steady-state reality field

### Phase 2: Visual Occlusion (30 seconds)  
- Cover LEFT camera only
- IMU continues normal operation
- RIGHT camera continues normal operation
- **Expected**: Trust in left camera should decrease

### Phase 3: Motion Conflict (30 seconds)
- Uncover LEFT camera
- Keep cameras pointed at static scene
- Actively move/rotate IMU (shake device)
- **Expected**: System detects conflict, adjusts weights

### Phase 4: Recovery (30 seconds)
- Return to normal operation
- All sensors providing consistent data
- **Expected**: Trust weights gradually recover

### Phase 5: Full Occlusion (30 seconds)
- Cover BOTH cameras
- IMU only input
- **Expected**: IMU trust increases, camera trust decreases

## Data Collection

### Logged Metrics (10Hz sampling)
```json
{
  "timestamp": "ISO-8601",
  "tick": 0,
  "phase": "baseline|occlusion|conflict|recovery|full_occlusion",
  "trust_weights": {
    "camera_left": 0.0-1.0,
    "camera_right": 0.0-1.0,
    "imu": 0.0-1.0
  },
  "sensor_readings": {
    "camera_motion": 0.0-1.0,
    "imu_stability": 0.0-1.0
  },
  "reality_field": 0.0-1.0,
  "context_state": "STABLE|MOVING|UNSTABLE|NOVEL",
  "conflict_detected": true|false
}
```

### Output Files
- `experiments/trust-dynamics/trust_log_YYYYMMDD_HHMMSS.jsonl` - Main data log
- `experiments/trust-dynamics/trust_analysis.json` - Post-experiment analysis
- `experiments/trust-dynamics/trust_plot.png` - Visualization of trust evolution

## Expected Results

1. **Trust Adaptation Speed**: 5-10 seconds to detect and adapt to sensor failure
2. **Trust Recovery**: 10-20 seconds to restore trust after sensor returns
3. **Conflict Resolution**: System should favor consistent sensors over conflicting ones
4. **Context Switching**: Expect STABLE → UNSTABLE during conflicts
5. **Reality Field**: Should remain stable despite sensor conflicts (robustness)

## Success Criteria

- [ ] Trust weights change by >20% during sensor conflicts
- [ ] System maintains coherent reality field despite conflicts  
- [ ] Trust recovery occurs within 30 seconds
- [ ] No system crashes during sensor manipulation
- [ ] Clear correlation between sensor reliability and trust

## Analysis Plan

1. Plot trust weights over time
2. Calculate trust adaptation rates
3. Measure reality field stability
4. Count context state transitions
5. Identify trust weight convergence patterns

## Safety Notes
- Handle cameras gently when covering/uncovering
- Don't shake IMU too violently (avoid damage)
- Monitor system resources during test
- Save data incrementally to prevent loss