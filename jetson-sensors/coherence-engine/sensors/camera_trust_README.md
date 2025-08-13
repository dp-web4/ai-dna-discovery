# Camera Trust Score — Quick Start (Jetson)

This doc shows how to use `camera_trust.py` on a Jetson (or any Linux box) to produce a fast **0–1 trust score** per camera frame.

## 1) Files

- `camera_trust.py` — scoring function (download it next to this README)
- (optional) `test_camera_trust.py` — tiny test harness (below)

## 2) Dependencies

Jetson + JetPack usually already includes OpenCV (`cv2`). If not (or on other hosts):

```bash
# Prefer system OpenCV on Jetson. If you need pip:
python3 -m pip install --upgrade pip
python3 -m pip install opencv-python-headless numpy
```

> If OpenCV is preinstalled via apt, you can skip pip entirely.

## 3) Minimal Usage (import)

```python
import cv2
from camera_trust import camera_trust_score

cap = cv2.VideoCapture(0)   # change index for other cams
prev_gray = None

while True:
    ok, bgr = cap.read()
    if not ok: break

    score, metrics = camera_trust_score(bgr, prev_gray=prev_gray, resize_w=320)

    # prepare prev_gray for temporal noise on next iteration
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    prev_gray = g

    print(f"trust={score:.3f}  edges={metrics['edge_density']:.3f}  contrast={metrics['rms_contrast']:.3f}")
```

## 4) CLI Test Harness

Save this as `test_camera_trust.py` next to `camera_trust.py`:

```python
import cv2, time, argparse
from camera_trust import camera_trust_score

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0, help="camera index")
ap.add_argument("--width", type=int, default=320, help="resize width for scoring")
ap.add_argument("--show", action="store_true", help="show live preview window")
args = ap.parse_args()

cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera index {args.cam}")

prev_gray = None

print("Press Ctrl+C to quit.")
try:
    while True:
        ok, frame = cap.read()
        if not ok: break

        score, m = camera_trust_score(frame, prev_gray=prev_gray, resize_w=args.width)

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = g

        print(f"score={score:.3f}  "
              f"sharp≈{m['tenengrad']:.1f}/{m['lap_var']:.1f}  "
              f"edges={m['edge_density']:.3f}  "
              f"contrast={m['rms_contrast']:.3f}  "
              f"sat={m['sat_mean']:.3f}  "
              f"clip[L/H]={m['low_clip']:.3f}/{m['high_clip']:.3f}  "
              f"noise={m['spatial_noise']:.3f}")

        if args.show:
            cv2.imshow("cam", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
```

Run it:

```bash
python3 test_camera_trust.py --cam 0            # headless log
python3 test_camera_trust.py --cam 0 --show     # if you have a display
```

## 5) Multi‑Camera Arbitration (soft routing)

Simple example to pick the top camera by trust with hysteresis:

```python
import cv2, numpy as np
from camera_trust import camera_trust_score

cams = [cv2.VideoCapture(i) for i in (0,1)]   # add more if needed
prev_g = [None]*len(cams)
ema = np.zeros(len(cams), dtype=np.float32)
alpha = 0.2      # EMA smoothing
stick = 0.05     # hysteresis margin

active = 0
while True:
    scores = []
    for i, cap in enumerate(cams):
        ok, bgr = cap.read()
        if not ok: scores.append(0.0); continue
        s, _ = camera_trust_score(bgr, prev_gray=prev_g[i], resize_w=320)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        prev_g[i] = g
        ema[i] = (1-alpha)*ema[i] + alpha*s
        scores.append(ema[i])

    best = int(np.argmax(scores))
    # hysteresis: only switch if best exceeds current by margin
    if scores[best] > scores[active] + stick:
        active = best

    # use `active` camera downstream
    print(f"active={active} scores={['%.3f'%x for x in scores]}")
```

## 6) Tuning Hints

- Change `resize_w` to trade off speed/accuracy (e.g., 240 or 320 is plenty).
- Adjust normalizer ranges in `camera_trust_score` for your optics/lighting:
  - `tenengrad` ≈ 40..300, `lap_var` ≈ 20..300 (at ~320px width)
  - `edge_density` ≈ 0.01..0.15
  - `rms_contrast` ≈ 0.05..0.25
  - `sat_mean` ≈ 0.15..0.75
  - clip fraction threshold ≈ 0.05..0.30
  - spatial noise ≈ 0.02..0.10
- Keep a per-camera EMA of trust for stability.
- If you have a motion mask, pass its mean into a custom temporal gate so motion isn’t penalized as “noise.”

## 7) Integrating with the Coherence Engine

Treat the trust as a **routing prior**:
- Use the score as per-camera relevance weight (normalize across cameras).
- Downstream, multiply feature vectors by this weight before fusion.
- Log the raw metrics alongside the fused output for sleep-cycle analysis.

---

**Files to copy:**
- `camera_trust.py`
- (optional) `test_camera_trust.py`

