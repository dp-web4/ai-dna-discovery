"""
Vision sensor plugin - corrected with instance handling
"""
import numpy as np
import time
from typing import Dict, Any, Optional
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from plugins.base_v2 import SensorBase

logger = logging.getLogger(__name__)

class VisionSensor(SensorBase):
    """Dual CSI camera sensor for Jetson"""
    
    def __init__(self, manifest: Dict[str, Any]):
        """Initialize with manifest"""
        super().__init__(manifest)
        self.resolution = None
        self.fps = None
        self.dual_cam = None
        self.device_ids = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        
    def initialize(self, config: Dict[str, Any]):
        """Initialize the vision sensor"""
        super().initialize(config)
        
        # Extract config
        self.resolution = tuple(config.get("resolution", [1920, 1080]))
        self.fps = config.get("fps", 30)
        self.dual_cam = config.get("dual_cam", True)
        self.device_ids = config.get("device_ids", [0, 1])
        
        logger.info(f"Initialized VisionSensor {self.lct}: {self.resolution}@{self.fps}fps")
        
        # In real implementation, initialize cameras here
        # self.cameras = [cv2.VideoCapture(id) for id in self.device_ids]
        
    def read(self) -> Dict[str, Any]:
        """Read frame data from cameras"""
        # Simulate frame capture with proper timing
        current_time = time.time()
        time_since_last = current_time - self.last_frame_time
        expected_interval = 1.0 / self.fps
        
        # Simple rate limiting
        if time_since_last < expected_interval:
            time.sleep(expected_interval - time_since_last)
        
        self.last_frame_time = time.time()
        self.frame_count += 1
        
        # Generate dummy frame data (in real impl, capture from cameras)
        frames = []
        for device_id in (self.device_ids if self.dual_cam else [self.device_ids[0]]):
            frame = np.random.randint(0, 255, 
                (*self.resolution, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Compute motion detection (dummy)
        motion = {
            "detected": bool(self.frame_count % 10 == 0),
            "regions": [] if self.frame_count % 10 else 
                      [{"x": 100, "y": 100, "w": 50, "h": 50}]
        }
        
        return {
            "frames": frames,
            "motion": motion,
            "timestamp": current_time,
            "frame_number": self.frame_count,
            "sensor_id": self.lct
        }
    
    def get_stereo_depth(self) -> Optional[np.ndarray]:
        """Compute stereo depth if dual cameras available"""
        if not self.dual_cam:
            return None
            
        # Dummy depth map
        depth = np.random.random(self.resolution) * 10  # 0-10 meters
        return depth
    
    def teardown(self):
        """Clean up camera resources"""
        logger.info(f"Shutting down VisionSensor {self.lct}")
        # In real impl: release cameras
        # for cam in self.cameras:
        #     cam.release()