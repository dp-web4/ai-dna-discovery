# Trust Dynamics Experiment Progress Report

*August 12, 2025*

## Executive Summary

Successfully implemented and tested camera trustworthiness assessment for the coherence engine. After multiple iterations and approaches, GPT's comprehensive camera trust scoring system proved effective at detecting occlusion and data quality issues.

## Journey Overview

### Initial Challenge
The coherence engine needed to assess whether visual sensor data is **trustworthy** - not just detecting occlusion, but determining if the data contains meaningful spatial/environmental information for reality field synthesis.

### Key Insight Evolution
1. Started with simple occlusion detection (brightness/darkness)
2. Realized camera quality isn't just about covering - it's about data trustworthiness
3. Shifted focus from binary covered/uncovered to continuous trust score
4. Discovered that normal CSI camera scenes have unexpected characteristics:
   - Edge ratio: 0.3+ (not 0.05 as initially assumed)
   - High variance in metrics even when stable
   - Auto-gain compensation when covered

## Approaches Tested

### V1: Trust Collapse Issue
- **Problem**: Trust weights dropped to zero and never recovered
- **Issue**: Multiplicative trust decay without proper recovery mechanism

### V2: Separate Camera Trust
- **Problem**: Camera quality always showed 1.0
- **Issue**: Edge detection thresholds too low (50/30 vs actual 100+/70+)
- **Lost**: IMU display disappeared

### V3: Canny Edge Detection
- **Problem**: Both cameras stuck at quality 0.5
- **Discovery**: Normal scenes have 0.3+ edge ratio, code expected <0.15
- **Issue**: Thresholds completely wrong for real camera data

### V4: Corrected Thresholds
- **Problem**: Quality barely changed between normal and covered (0.80 → 0.84)
- **Issue**: Even with "corrected" thresholds, metrics weren't detecting occlusion

### Final Solution: GPT's Camera Trust
- **Success**: Trust drops from ~0.73 to ~0.47 when covered
- **Recovery**: Returns to ~0.73 when uncovered
- **Comprehensive**: Multiple signals combined with proper weighting

## Technical Discoveries

### Camera Characteristics (Jetson CSI)
```
Normal Scene:
- Edge ratio: 0.32 (320x range)
- Brightness: 80-100
- Contrast: 60-80 (std dev)
- Laplacian variance: 4000-5000
- Tenengrad: 4200-4500

Covered Camera:
- Edge ratio: <0.001
- Brightness: 70-80 (auto-gain compensates!)
- Contrast: 70-80 (noise maintains contrast!)
- Laplacian variance: 20000+ (noise amplified)
- Tenengrad: 5000-6000
```

**Critical Finding**: When covered, the camera doesn't go dark - auto-gain and noise actually INCREASE some metrics!

### Why Simple Approaches Failed

1. **Brightness doesn't drop** - Auto-gain keeps image bright even when covered
2. **Contrast stays high** - Noise maintains standard deviation
3. **Edges can increase** - Noise creates false edges
4. **Focus metrics misleading** - Laplacian variance increases with noise

### GPT's Solution Components

The successful approach combines:
- **Sharpness** (Tenengrad + Laplacian) - 30% weight
- **Edge density** (Canny on blurred image) - 15% weight
- **RMS contrast** (normalized std dev) - 15% weight
- **Saturation** (HSV color richness) - 10% weight
- **Exposure clipping** (over/under exposure) - 15% weight
- **Noise estimation** (spatial + temporal) - 15% weight

Each metric is:
1. Normalized to 0-1 range with calibrated thresholds
2. Passed through sigmoid for soft squashing
3. Combined with weights
4. Returns continuous trust score

## Integration Path

### Current Status
- ✅ Camera trust scoring working standalone
- ✅ Detects occlusion reliably
- ✅ Provides continuous 0-1 trust score
- ✅ Fast enough for real-time (< 10ms per frame)

### Next Steps
1. Integrate into coherence engine visual sensor
2. Use trust score to weight camera contribution to reality field
3. Implement per-camera trust evolution over time
4. Add multi-camera arbitration with hysteresis

## Lessons Learned

### What Doesn't Work
- Simple threshold-based detection
- Relying on single metrics (brightness, edges, etc.)
- Assuming covered = dark
- Multiplicative quality scores (one bad metric → zero)

### What Works
- Multiple complementary signals
- Weighted combination (not multiplication)
- Sigmoid squashing to prevent dominance
- Calibrated thresholds for specific hardware
- Temporal analysis for noise detection

## Code Organization

```
experiments/trust-dynamics/
├── EXPERIMENT_PLAN.md              # Original 5-phase plan
├── CAMERA_QUALITY_DESIGN.md        # Design before implementation
├── coherence_with_trust_v*.py      # Evolution attempts (V1-V4)
├── camera_quality_*.py              # Various quality detection attempts
├── test_gpt_camera_trust.py        # Successful GPT implementation test
└── TRUST_DYNAMICS_PROGRESS.md      # This document

coherence-engine/
├── camera_trust.py                 # GPT's working implementation
└── camera_trust_README.md          # GPT's documentation
```

## Performance Metrics

- **Processing time**: ~8ms per frame at 320px width
- **Trust score range**: 0.47-0.74 (good discrimination)
- **Recovery time**: < 1 second after uncovering
- **Stability**: ±0.001 std dev during stable conditions

## Philosophical Reflection

This journey revealed that "trust" in sensor data isn't binary - it's a continuous assessment of information quality. A covered camera isn't just "off" - it's providing untrustworthy data that looks superficially valid. The coherence engine needs to continuously evaluate not just what sensors report, but how much to trust those reports.

The challenge wasn't detecting darkness - it was detecting **meaninglessness** in data that appears normal.

## Conclusion

After extensive experimentation with multiple approaches, we have a working camera trustworthiness system. GPT's implementation successfully detects when camera data is unreliable, providing the coherence engine with the signal quality assessment needed for robust sensor fusion.

The trust dynamics system can now:
- Detect camera occlusion/degradation
- Provide continuous trust scores
- Recover quickly from temporary issues
- Weight sensor contributions appropriately

Ready for integration into the main coherence engine.

---

*"Trust isn't given - it's computed, frame by frame, metric by metric."*