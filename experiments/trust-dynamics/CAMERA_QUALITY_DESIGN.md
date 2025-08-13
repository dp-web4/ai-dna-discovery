# Camera Quality Detection Design

## Goal
Detect when a camera is providing useful visual data vs when it's occluded/covered/degraded.

## Key Observations from Testing
1. **Normal scene**: Edge ratio ~0.32, Brightness 80-100, Contrast 60-80, Focus 20k-120k
2. **Covered camera**: Edge ratio <0.01, Brightness <20, Contrast <20, Focus <5k
3. **The most reliable indicator of occlusion is darkness (brightness < 20)**

## Metrics to Extract

### 1. Brightness (Most Reliable)
- **Tool**: `np.mean(grayscale_image)`
- **Normal Range**: 50-150
- **Covered Range**: 0-20
- **Calculation**: 
  - If brightness < 20: quality drops linearly to 0
  - If brightness 20-50: quality ramps from 0.5 to 1.0
  - If brightness 50-150: quality = 1.0
  - If brightness > 150: slight penalty for overexposure

### 2. Contrast (Second Most Reliable)
- **Tool**: `np.std(grayscale_image)`
- **Normal Range**: 40-100
- **Covered Range**: 0-20
- **Calculation**:
  - If contrast < 10: quality drops to near 0 (uniform image)
  - If contrast 10-40: quality ramps from 0.3 to 1.0
  - If contrast > 40: quality = 1.0

### 3. Edge Presence (Supporting Metric)
- **Tool**: `cv2.Canny(gray, 50, 150)` then count edge pixels
- **Normal Range**: 0.25-0.35 ratio
- **Covered Range**: < 0.01 ratio
- **Calculation**:
  - If edge_ratio < 0.01: strong indicator of occlusion
  - If edge_ratio 0.01-0.1: partial occlusion possible
  - If edge_ratio > 0.1: likely normal

### 4. Sharpness/Focus (Least Reliable)
- **Tool**: `cv2.Laplacian(gray, cv2.CV_64F).var()`
- **Normal Range**: 10k-200k (highly variable)
- **Covered Range**: < 1k
- **Note**: High variance, depends on scene content

## Quality Score Calculation

### Approach: Weighted Average with Thresholds
```
quality = weighted_average(
    brightness_score * 0.4,  # Most weight
    contrast_score * 0.3,
    edge_score * 0.2,
    focus_score * 0.1  # Least weight
)
```

### Early Exit Conditions
If any of these are true, immediately return low quality:
- brightness < 10 → quality = brightness/10 (max 0.1)
- contrast < 5 → quality = contrast/5 (max 0.1)

## Implementation Strategy

1. **Calculate raw metrics** from the frame
2. **Apply early exit** checks for obvious occlusion
3. **Convert each metric to 0-1 score** using appropriate thresholds
4. **Combine scores** using weighted average
5. **Return final quality** (0.0 = covered, 1.0 = normal)

## Testing Protocol
1. Start with normal scene → should show 0.8-1.0
2. Cover camera completely → should drop to 0.0-0.2
3. Partially cover → should show 0.3-0.7
4. Uncover → should return to 0.8-1.0

## Key Insight
**Don't overthink it**: A covered camera is primarily DARK. Focus on brightness first, use other metrics as confirmation.