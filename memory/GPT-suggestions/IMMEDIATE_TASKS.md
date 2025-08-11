# 🔧 Immediate Implementation Tasks

## 1. IMU Mount Optimization
- Build and deploy a physical horizontal mounting bracket.
- Modify `imu_vertical_confidence.py` to detect mount orientation and apply remapping.
- Restore real-time heading confidence with the corrected vertical axis.

## 2. Stereo Camera Enhancement
- Add real-time calibration quality metrics (e.g. reprojection error).
- Integrate OpenCV stereo matching confidence into `StereoCameraConfidence`.
- Optionally use brightness histograms or light sensors to adjust expectations.

## 3. Temporal Memory Buffers
- Add optional ring buffers per sensor for raw confidence data.
- Start modeling confidence decay and stability patterns over time.
