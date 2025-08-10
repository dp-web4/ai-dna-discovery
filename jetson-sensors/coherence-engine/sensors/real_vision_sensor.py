"""
Real vision sensor adapter for CSI cameras on Jetson.
Bridges our working stereo vision to the coherence engine Protocol.
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple
import logging

logger = logging.getLogger("real_vision")

@dataclass
class RealVisionSensor:
    """
    Adapts dual CSI cameras to coherence engine sensor Protocol.
    Returns confidence score based on motion detection and stereo correlation.
    """
    id: str = "vision"
    camera_width: int = 640
    camera_height: int = 480
    motion_threshold: float = 30.0
    
    # Camera state
    cap_l: Optional[cv2.VideoCapture] = field(default=None, init=False)
    cap_r: Optional[cv2.VideoCapture] = field(default=None, init=False)
    prev_gray_l: Optional[np.ndarray] = field(default=None, init=False)
    prev_gray_r: Optional[np.ndarray] = field(default=None, init=False)
    initialized: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Initialize CSI cameras."""
        try:
            # GStreamer pipeline for CSI cameras
            def gst_pipeline(sensor_id=0):
                return (
                    f"nvarguscamerasrc sensor-id={sensor_id} ! "
                    f"video/x-raw(memory:NVMM), width=3280, height=2464, format=NV12, framerate=21/1 ! "
                    f"nvvidconv ! video/x-raw, width={self.camera_width}, height={self.camera_height}, format=BGRx ! "
                    f"videoconvert ! video/x-raw, format=BGR ! appsink"
                )
            
            self.cap_l = cv2.VideoCapture(gst_pipeline(0), cv2.CAP_GSTREAMER)
            self.cap_r = cv2.VideoCapture(gst_pipeline(1), cv2.CAP_GSTREAMER)
            
            # Test read
            ret_l, _ = self.cap_l.read()
            ret_r, _ = self.cap_r.read()
            
            if ret_l and ret_r:
                self.initialized = True
                logger.info("Real vision sensor initialized with dual CSI cameras")
            else:
                logger.warning("Camera initialization failed, falling back to simulation")
                
        except Exception as e:
            logger.warning(f"Could not initialize cameras: {e}, using simulation mode")
            
    def read(self, *, tick: int) -> float:
        """
        Read from real cameras and return confidence score [0,1].
        Confidence based on:
        - Motion detection in both cameras
        - Stereo correspondence between views
        - Frame capture success rate
        """
        if not self.initialized:
            # Fallback to simulated value
            import math
            return 0.5 + 0.3 * math.sin(tick / 10.0)
            
        try:
            ret_l, frame_l = self.cap_l.read()
            ret_r, frame_r = self.cap_r.read()
            
            if not (ret_l and ret_r):
                return 0.1  # Low confidence if capture fails
                
            # Convert to grayscale for processing
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            
            confidence = 0.5  # Base confidence for successful capture
            
            # Motion detection
            if self.prev_gray_l is not None and self.prev_gray_r is not None:
                # Optical flow for motion
                flow_l = cv2.calcOpticalFlowFarneback(
                    self.prev_gray_l, gray_l, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                flow_r = cv2.calcOpticalFlowFarneback(
                    self.prev_gray_r, gray_r, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Motion magnitude
                mag_l = np.sqrt(flow_l[..., 0]**2 + flow_l[..., 1]**2)
                mag_r = np.sqrt(flow_r[..., 0]**2 + flow_r[..., 1]**2)
                
                motion_l = np.mean(mag_l)
                motion_r = np.mean(mag_r)
                
                # Higher confidence when motion is detected consistently
                if motion_l > 0.5 and motion_r > 0.5:
                    motion_agreement = 1.0 - abs(motion_l - motion_r) / max(motion_l, motion_r)
                    confidence += 0.3 * motion_agreement
                    
            # Stereo correspondence check
            # Simple correlation between left and right images
            if gray_l.shape == gray_r.shape:
                correlation = np.corrcoef(gray_l.flatten(), gray_r.flatten())[0, 1]
                # Higher confidence when images are somewhat correlated (stereo pair)
                # but not identical (would indicate camera failure)
                if 0.3 < correlation < 0.9:
                    confidence += 0.2 * correlation
                    
            # Store for next frame
            self.prev_gray_l = gray_l
            self.prev_gray_r = gray_r
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Vision sensor read error: {e}")
            return 0.0
            
    def __del__(self):
        """Clean up camera resources."""
        if self.cap_l:
            self.cap_l.release()
        if self.cap_r:
            self.cap_r.release()