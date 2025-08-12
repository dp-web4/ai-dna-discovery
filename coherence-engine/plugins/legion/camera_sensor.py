"""
Camera sensor for Legion - uses webcam or USB cameras via OpenCV
"""
import threading
import time
from typing import Optional, Tuple

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - camera sensor disabled")

class CameraSensor:
    """Webcam/USB camera sensor for Legion"""
    
    def __init__(self, camera_index: int = 0):
        self.id = "camera"
        self.camera_index = camera_index
        self.available = False
        self.motion_level = 0.0
        self.last_frame = None
        self.running = False
        self.thread = None
        
        self.setup_camera()
    
    def setup_camera(self):
        """Initialize camera capture"""
        if not CV2_AVAILABLE:
            self.available = False
            return
            
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            # Try to set reasonable resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test if camera works
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.available = True
                self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Start background thread for motion detection
                self.running = True
                self.thread = threading.Thread(target=self.motion_monitor_loop, daemon=True)
                self.thread.start()
            else:
                print(f"Camera {self.camera_index} not available")
                self.cap.release()
                
        except Exception as e:
            print(f"Camera setup failed: {e}")
            self.available = False
    
    def motion_monitor_loop(self):
        """Background thread to monitor motion"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate motion using frame difference
                    if self.last_frame is not None:
                        # Frame difference
                        diff = cv2.absdiff(self.last_frame, gray)
                        
                        # Threshold to get binary motion mask
                        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        
                        # Calculate motion percentage
                        motion_pixels = np.count_nonzero(motion_mask)
                        total_pixels = motion_mask.shape[0] * motion_mask.shape[1]
                        motion_percent = motion_pixels / total_pixels
                        
                        # Alternative: Use Laplacian for focus/detail detection
                        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
                        lap_normalized = min(1.0, lap / 1000.0)
                        
                        # Combine motion and detail
                        self.motion_level = motion_percent * 0.5 + lap_normalized * 0.5
                    
                    self.last_frame = gray
                    
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                if self.running:
                    print(f"Camera read error: {e}")
                time.sleep(0.1)
    
    def read(self, *, tick: int) -> float:
        """
        Read camera sensor data.
        Returns normalized [0,1] value based on detected motion/activity.
        """
        if not self.available:
            return 0.0
        
        return self.motion_level
    
    def get_current_frame(self):
        """Get the current camera frame for visualization
        
        Returns:
            Optional numpy array if camera available and CV2 imported, None otherwise
        """
        if not self.available or not CV2_AVAILABLE:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def get_camera_info(self) -> dict:
        """Get camera information"""
        if not self.available:
            return {"available": False, "index": self.camera_index}
        
        return {
            "available": True,
            "index": self.camera_index,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS))
        }
    
    def cleanup(self):
        """Clean up camera resources"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if hasattr(self, 'cap'):
            self.cap.release()