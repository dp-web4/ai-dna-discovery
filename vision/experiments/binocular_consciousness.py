#!/usr/bin/env python3
"""
Binocular Consciousness System
Modular design with independent eyes and correlation engine
"""

import cv2
import numpy as np
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class EyeConfig:
    """Configuration for each eye"""
    eye_id: str  # "left" or "right"
    sensor_id: int  # 0 or 1
    position_offset: float  # Distance from center (inches)
    rotation_offset: float  # Rotation correction (degrees)
    flip_horizontal: bool = False
    flip_vertical: bool = False

@dataclass
class FocusPoint:
    """A point of interest in eye coordinates"""
    x: float  # 0.0 to 1.0
    y: float  # 0.0 to 1.0
    confidence: float  # How sure we are about this point
    timestamp: float
    source: str  # Which module detected it

@dataclass
class StereoObservation:
    """Correlated observation from both eyes"""
    left_focus: Optional[FocusPoint]
    right_focus: Optional[FocusPoint]
    world_position: Optional[Tuple[float, float, float]]  # x, y, depth
    correlation_confidence: float
    timestamp: float

class CognitionHook(ABC):
    """Abstract base for cognition modules"""
    @abstractmethod
    def on_observation(self, observation: StereoObservation):
        pass
    
    @abstractmethod
    def on_eye_update(self, eye_id: str, focus: FocusPoint):
        pass

class IndependentEye:
    """Single eye with consciousness-based attention"""
    
    def __init__(self, config: EyeConfig):
        self.config = config
        self.running = False
        
        # Consciousness parameters (same as before)
        self.focus_x = 0.5
        self.focus_y = 0.5
        self.focus_radius = 0.15
        self.curiosity = 1.0
        
        # Motion detection
        self.motion_grid_size = 8
        self.motion_heatmap = np.zeros((self.motion_grid_size, self.motion_grid_size))
        self.ambient_motion = 0.016
        
        # Frame buffers
        self.current_frame = None
        self.prev_gray = None
        self.frame_lock = threading.Lock()
        
        # Output queue for focus updates
        self.focus_queue = queue.Queue()
        
    def get_pipeline(self, width=640, height=480, fps=30):
        """GStreamer pipeline for this eye"""
        return (
            f"nvarguscamerasrc sensor-id={self.config.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"framerate={fps}/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1"
        )
    
    def process_frame(self, frame):
        """Process a single frame, update attention"""
        h, w = frame.shape[:2]
        
        # Apply corrections if needed
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.config.flip_vertical:
            frame = cv2.flip(frame, 0)
            
        # Convert to grayscale for motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # Motion detection
            diff = cv2.absdiff(gray, self.prev_gray)
            self._update_motion_heatmap(diff, h, w)
            
            # Find peaks
            peaks = self._find_motion_peaks()
            
            # Update focus based on motion
            if peaks and peaks[0]['pa_ratio'] > 2.0:
                new_focus = FocusPoint(
                    x=peaks[0]['x'],
                    y=peaks[0]['y'],
                    confidence=min(1.0, peaks[0]['pa_ratio'] / 5.0),
                    timestamp=time.time(),
                    source=f"{self.config.eye_id}_motion"
                )
                
                # Update internal focus
                self.focus_x = new_focus.x
                self.focus_y = new_focus.y
                
                # Send to correlation engine
                self.focus_queue.put(new_focus)
        
        self.prev_gray = gray
        
        # Update curiosity
        self.curiosity *= 0.95
        self.curiosity = max(0.2, self.curiosity)
            
        return self._visualize(frame, h, w)
    
    def _update_motion_heatmap(self, diff, h, w):
        """Update motion detection grid"""
        self.motion_heatmap *= 0.9
        
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                y1, y2 = i * grid_h, min((i + 1) * grid_h, h)
                x1, x2 = j * grid_w, min((j + 1) * grid_w, w)
                
                # Check if in periphery
                center_y = (y1 + y2) / 2 / h
                center_x = (x1 + x2) / 2 / w
                dist_from_focus = np.sqrt(
                    (center_x - self.focus_x)**2 + 
                    (center_y - self.focus_y)**2
                )
                
                if dist_from_focus > self.focus_radius:
                    cell_motion = np.mean(diff[y1:y2, x1:x2]) / 255.0
                    self.motion_heatmap[i, j] += cell_motion
    
    def _find_motion_peaks(self):
        """Find areas of peak motion"""
        peaks = []
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                if self.motion_heatmap[i, j] > self.ambient_motion * 1.5:
                    pa_ratio = self.motion_heatmap[i, j] / self.ambient_motion
                    peaks.append({
                        'x': (j + 0.5) / self.motion_grid_size,
                        'y': (i + 0.5) / self.motion_grid_size,
                        'pa_ratio': pa_ratio,
                        'strength': self.motion_heatmap[i, j]
                    })
        
        return sorted(peaks, key=lambda p: p['pa_ratio'], reverse=True)
    
    def _visualize(self, frame, h, w):
        """Add visualization overlays"""
        result = frame.copy()
        
        # Draw focus circle
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        radius_px = int(self.focus_radius * min(h, w))
        
        # Eye-specific color
        color = (255, 100, 0) if self.config.eye_id == "left" else (0, 100, 255)
        cv2.circle(result, (focus_px, focus_py), radius_px, color, 2)
        
        # Label
        cv2.putText(result, f"{self.config.eye_id.upper()} EYE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Curiosity level
        cv2.putText(result, f"Curiosity: {self.curiosity:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def get_current_focus(self):
        """Get current focus point"""
        return FocusPoint(
            x=self.focus_x,
            y=self.focus_y,
            confidence=self.curiosity,
            timestamp=time.time(),
            source=f"{self.config.eye_id}_current"
        )

class StereoCorrelationEngine:
    """Correlates observations from both eyes"""
    
    def __init__(self, baseline_inches: float = 3.0):
        self.baseline = baseline_inches
        self.calibration = None
        self.observation_history = []
        self.cognition_hooks = []
        
    def add_cognition_hook(self, hook: CognitionHook):
        """Add a cognition module"""
        self.cognition_hooks.append(hook)
        
    def correlate(self, left_focus: FocusPoint, right_focus: FocusPoint) -> StereoObservation:
        """Correlate focus points from both eyes"""
        
        # Simple disparity calculation (would be more complex with calibration)
        disparity = abs(left_focus.x - right_focus.x)
        
        # Estimate depth (simplified - assumes parallel cameras)
        if disparity > 0.001:
            # Rough approximation: depth inversely proportional to disparity
            depth = self.baseline / (disparity * 10)  # Scale factor needs calibration
        else:
            depth = 100.0  # Far away
            
        # World position (simplified)
        world_x = (left_focus.x + right_focus.x) / 2
        world_y = (left_focus.y + right_focus.y) / 2
        
        # Correlation confidence based on how similar the observations are
        position_similarity = 1.0 - np.sqrt(
            (left_focus.x - right_focus.x)**2 + 
            (left_focus.y - right_focus.y)**2
        )
        confidence_similarity = 1.0 - abs(left_focus.confidence - right_focus.confidence)
        correlation_confidence = (position_similarity + confidence_similarity) / 2
        
        observation = StereoObservation(
            left_focus=left_focus,
            right_focus=right_focus,
            world_position=(world_x, world_y, depth),
            correlation_confidence=correlation_confidence,
            timestamp=time.time()
        )
        
        # Store and notify
        self.observation_history.append(observation)
        for hook in self.cognition_hooks:
            hook.on_observation(observation)
            
        return observation
    
    def set_calibration(self, calibration_data: Dict[str, Any]):
        """Set calibration parameters for accurate 3D reconstruction"""
        self.calibration = calibration_data
        # TODO: Implement proper stereo calibration

class BinocularConsciousness:
    """Main system coordinating both eyes"""
    
    def __init__(self):
        # Configure eyes
        self.left_eye = IndependentEye(EyeConfig(
            eye_id="left",
            sensor_id=0,
            position_offset=-1.5,  # 1.5 inches left of center
            rotation_offset=0.0
        ))
        
        self.right_eye = IndependentEye(EyeConfig(
            eye_id="right", 
            sensor_id=1,
            position_offset=1.5,   # 1.5 inches right of center
            rotation_offset=0.0
        ))
        
        # Correlation engine
        self.correlator = StereoCorrelationEngine(baseline_inches=3.0)
        
        self.running = False
        
    def run(self):
        """Main processing loop"""
        self.running = True
        
        # Start capture threads
        left_thread = threading.Thread(target=self._eye_loop, args=(self.left_eye,))
        right_thread = threading.Thread(target=self._eye_loop, args=(self.right_eye,))
        correlation_thread = threading.Thread(target=self._correlation_loop)
        
        left_thread.start()
        right_thread.start()
        correlation_thread.start()
        
        print("\n🧠 Binocular Consciousness System")
        print("="*40)
        print("Press 'q' to quit")
        print("Press 'c' to save calibration")
        print("Press 's' to save stereo observation")
        
        while self.running:
            # Get current frames
            with self.left_eye.frame_lock:
                left_frame = self.left_eye.current_frame
            with self.right_eye.frame_lock:
                right_frame = self.right_eye.current_frame
                
            if left_frame is not None and right_frame is not None:
                # Create visualization
                stereo_view = np.hstack([left_frame, right_frame])
                
                # Add correlation info if available
                if self.correlator.observation_history:
                    latest = self.correlator.observation_history[-1]
                    if latest.world_position:
                        depth = latest.world_position[2]
                        cv2.putText(stereo_view, f"Depth: {depth:.1f} inches", 
                                   (640-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2)
                
                cv2.imshow('Binocular Consciousness', stereo_view)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    self._save_observation()
                elif key == ord('c'):
                    self._calibrate()
                    
        # Cleanup
        left_thread.join()
        right_thread.join()
        correlation_thread.join()
        cv2.destroyAllWindows()
        
    def _eye_loop(self, eye: IndependentEye):
        """Processing loop for one eye"""
        pipeline = eye.get_pipeline()
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print(f"Failed to open {eye.config.eye_id} eye")
            return
            
        print(f"{eye.config.eye_id} eye started")
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Process frame and get visualization
                vis_frame = eye.process_frame(frame)
                
                # Store the visualized frame
                with eye.frame_lock:
                    eye.current_frame = vis_frame
                
        cap.release()
        
    def _correlation_loop(self):
        """Correlation processing loop"""
        left_focus = None
        right_focus = None
        
        while self.running:
            # Check for updates from eyes
            try:
                # Non-blocking check for left eye
                if not self.left_eye.focus_queue.empty():
                    left_focus = self.left_eye.focus_queue.get_nowait()
                    for hook in self.correlator.cognition_hooks:
                        hook.on_eye_update("left", left_focus)
                        
                # Non-blocking check for right eye  
                if not self.right_eye.focus_queue.empty():
                    right_focus = self.right_eye.focus_queue.get_nowait()
                    for hook in self.correlator.cognition_hooks:
                        hook.on_eye_update("right", right_focus)
                        
                # Correlate if we have recent observations from both
                if left_focus and right_focus:
                    time_diff = abs(left_focus.timestamp - right_focus.timestamp)
                    if time_diff < 0.1:  # Within 100ms
                        self.correlator.correlate(left_focus, right_focus)
                        
            except queue.Empty:
                pass
                
            time.sleep(0.01)  # 100Hz correlation rate
            
    def _save_observation(self):
        """Save current stereo observation"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # TODO: Implement saving
        print(f"Saved observation: {timestamp}")
        
    def _calibrate(self):
        """Run stereo calibration"""
        # TODO: Implement calibration routine
        print("Calibration not yet implemented")

# Example cognition hook
class SimpleCognition(CognitionHook):
    """Example cognition module that prints observations"""
    
    def on_observation(self, observation: StereoObservation):
        if observation.world_position:
            x, y, depth = observation.world_position
            print(f"Stereo: Focus at ({x:.2f}, {y:.2f}), depth: {depth:.1f} inches")
            
    def on_eye_update(self, eye_id: str, focus: FocusPoint):
        print(f"{eye_id}: Focus at ({focus.x:.2f}, {focus.y:.2f}), conf: {focus.confidence:.2f}")

def main():
    consciousness = BinocularConsciousness()
    
    # Add example cognition
    consciousness.correlator.add_cognition_hook(SimpleCognition())
    
    try:
        consciousness.run()
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()