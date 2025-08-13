
import cv2
import numpy as np
from typing import Dict, Tuple, Optional

def _normalize(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def _sigmoid(x: float, k: float = 8.0) -> float:
    # squashes [0,1] with more weight near the center
    x = np.clip(x, 0.0, 1.0)
    return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))

def camera_trust_score(
    frame_bgr: np.ndarray,
    prev_gray: Optional[np.ndarray] = None,
    resize_w: int = 320,
) -> Tuple[float, Dict[str, float]]:
    '''
    Compute a fast 0..1 trustworthiness score for a camera frame.
    Signals:
      - sharpness (Tenengrad / Laplacian var)
      - edge density (Canny)
      - RMS contrast (std of gray)
      - saturation (mean S channel)
      - exposure clipping (fractions near 0/255)
      - noise (high-pass residual energy; optional temporal residual if prev provided)

    Returns:
      score, metrics_dict

    Notes:
      * Works best when called on downscaled frames (default width ~320 px).
      * All metrics are intentionally simple for Jetson-friendly speed.
    '''
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0, {"error": 1.0}

    h, w = frame_bgr.shape[:2]
    if resize_w and w > resize_w:
        scale = resize_w / float(w)
        frame_bgr = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # Convert
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # --- Sharpness / focus ---
    # Tenengrad: mean squared gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx*gx + gy*gy))  # typical ~ 50..300 on 320w
    sharp_ten = _normalize(tenengrad, 40.0, 300.0)

    # Laplacian variance (also correlates with blur)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))
    sharp_lap = _normalize(lap_var, 20.0, 300.0)

    sharpness = 0.6*sharp_ten + 0.4*sharp_lap

    # --- Edge density ---
    # Use Canny on blurred image for stability
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3,3), 0), 60, 120, L2gradient=True)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    # Favor moderate to high edge density; too low = blank/blur, too high often noise
    edge_score = _normalize(edge_density, 0.01, 0.15)

    # --- Contrast (RMS) ---
    rms_contrast = float(np.std(gray) / 255.0)  # ~0.05..0.25 typical
    contrast_score = _normalize(rms_contrast, 0.05, 0.25)

    # --- Saturation ---
    sat_mean = float(np.mean(hsv[...,1]) / 255.0)  # 0..1
    # mid-to-high saturation is generally good; too low suggests gray/washed
    saturation_score = _normalize(sat_mean, 0.15, 0.75)

    # --- Exposure clipping ---
    # fraction of pixels near black / near white
    low_clip  = float((gray < 10).mean())
    high_clip = float((gray > 245).mean())
    clip_frac = max(low_clip, high_clip)
    exposure_score = 1.0 - _normalize(clip_frac, 0.05, 0.30)  # tolerate up to ~5% clipped, punish >30%

    # --- Noise estimate ---
    # Spatial: high-pass residual energy (gray - blur)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    residual = gray.astype(np.float32) - blur.astype(np.float32)
    spatial_noise = float(np.std(residual) / 255.0)  # ~0.01..0.08 common
    spatial_noise_good = 1.0 - _normalize(spatial_noise, 0.02, 0.10)

    # Temporal: if prev_gray provided, difference energy (motion + noise). We invert but cap.
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = (gray.astype(np.int16) - prev_gray.astype(np.int16))
        td = float(np.std(diff) / 255.0)
        # Very high temporal diff may indicate motion; do not over-penalize.
        temporal_noise_good = 1.0 - _normalize(td, 0.02, 0.25)
        noise_good = 0.6*spatial_noise_good + 0.4*temporal_noise_good
    else:
        noise_good = spatial_noise_good

    # --- Combine with weights ---
    # You can tune these per camera class; they sum to 1.0 here.
    w = {
        "sharpness": 0.30,
        "edges":     0.15,
        "contrast":  0.15,
        "saturation":0.10,
        "exposure":  0.15,
        "noise":     0.15,
    }

    # softly squash each to avoid one metric dominating
    comps = {
        "sharpness": _sigmoid(sharpness),
        "edges":     _sigmoid(edge_score),
        "contrast":  _sigmoid(contrast_score),
        "saturation":_sigmoid(saturation_score),
        "exposure":  _sigmoid(exposure_score),
        "noise":     _sigmoid(noise_good),
    }

    score = sum(w[k]*comps[k] for k in w.keys())

    metrics = dict(
        score=score,
        tenengrad=tenengrad,
        lap_var=lap_var,
        edge_density=edge_density,
        rms_contrast=rms_contrast,
        sat_mean=sat_mean,
        low_clip=low_clip,
        high_clip=high_clip,
        spatial_noise=spatial_noise,
    )
    return float(score), metrics
