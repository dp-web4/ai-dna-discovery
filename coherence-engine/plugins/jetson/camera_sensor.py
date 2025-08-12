"""
Camera Sensor Plugin for Coherence Engine
Integrates dual CSI cameras with real GStreamer pipelines
August 12, 2025
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional
import threading
import queue

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.base import SensorBase

class CameraSensorPlugin(SensorBase):
    """Real dual CSI camera sensor plugin for Jetson"""
    
    def __init__(self, identity: str = "camera_sensor"):
        super().__init__(identity)
        self.caps = [None, None]
        self.latest_frames = [None, None]
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        self.running = False
        self.fps = 30
        self.resolution = (1920, 1080)
        self.display_resolution = (960, 540)  # For dashboard display
        
    def initialize(self, config: Dict[str, Any]):
        """Initialize dual CSI cameras with GStreamer"""
        self.fps = config.get("fps", 30)
        self.resolution = config.get("resolution", (1920, 1080))
        
        print(f"Initializing camera sensor: {self.resolution} @ {self.fps}fps")
        
        # Initialize both cameras
        for sensor_id in range(2):
            pipeline = self._gst_pipeline(sensor_id)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if cap.isOpened():
                self.caps[sensor_id] = cap
                print(f"✓ Camera {sensor_id} initialized")
            else:
                print(f"✗ Failed to initialize camera {sensor_id}")
                
        # Start capture thread
        if any(self.caps):
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
    def teardown(self):
        """Clean up camera resources"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            
        for i, cap in enumerate(self.caps):
            if cap:
                cap.release()
                self.caps[i] = None
                
        print("Camera sensor shutdown complete")
        
    def _gst_pipeline(self, sensor_id: int) -> str:
        """Create GStreamer pipeline for CSI camera"""
        width, height = self.resolution
        
        # Use sensor-mode=2 for 1920x1080 @ 30fps
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv ! video/x-raw, width={self.display_resolution[0]}, "
            f"height={self.display_resolution[1]}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        
    def _capture_loop(self):
        """Background thread for continuous capture"""
        while self.running:
            frames = []
            
            for i, cap in enumerate(self.caps):
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.latest_frames[i] = frame
                        frames.append(frame)
                    else:
                        frames.append(None)
                else:
                    frames.append(None)
                    
            # Put frames in queue for processing
            try:
                self.frame_queue.put_nowait(frames)
            except queue.Full:
                # Drop frame if queue is full
                pass
                
            # Maintain frame rate
            time.sleep(1.0 / self.fps)
            
    def read(self) -> Dict[str, Any]:
        """Read current camera data"""
        frames = self.latest_frames.copy()
        
        # Compute basic vision metrics
        motion = 0.0
        brightness = 0.0
        contrast = 0.0
        
        valid_frames = [f for f in frames if f is not None]
        
        if valid_frames:
            # Calculate average brightness
            for frame in valid_frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness += gray.mean()
                
                # Simple motion detection via Laplacian
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                motion += lap.var()
                
                # Contrast via standard deviation
                contrast += gray.std()
                
            num_frames = len(valid_frames)
            brightness /= (num_frames * 255.0)  # Normalize to 0-1
            motion = min(motion / (num_frames * 1000.0), 1.0)  # Normalize
            contrast /= (num_frames * 128.0)  # Normalize
            
        # Stereo disparity if both cameras available
        disparity = None
        if len(valid_frames) == 2:
            disparity = self._compute_simple_disparity(valid_frames[0], valid_frames[1])
            
        return {
            "frames": frames,
            "brightness": brightness,
            "motion": motion,
            "contrast": contrast,
            "disparity": disparity,
            "timestamp": time.time(),
            "confidence": 0.9 if valid_frames else 0.0
        }
        
    def _compute_simple_disparity(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute simple disparity metric between stereo frames"""
        # Convert to grayscale
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        
        # Compute difference
        diff = cv2.absdiff(gray_l, gray_r)
        
        # Return normalized disparity metric
        return diff.mean() / 255.0
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Declare camera sensor capabilities"""
        return {
            "type": "vision",
            "subtype": "dual_csi_camera",
            "resolution": self.resolution,
            "display_resolution": self.display_resolution,
            "fps": self.fps,
            "num_cameras": 2,
            "features": ["stereo_vision", "motion_detection", "brightness_sensing"],
            "confidence_range": [0.0, 1.0]
        }