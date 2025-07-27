#!/usr/bin/env python3
"""
Auto-Calibrating Binocular Vision V2
Combines working contour-based tracking with auto-calibration
"""

import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass
import json
import os

from binocular_consciousness import (
    EyeConfig, FocusPoint, StereoObservation, 
    CognitionHook, StereoCorrelationEngine
)

class AutoCalibratingEyeV2:
    """Eye with working motion tracking and auto-calibration"""
    
    def __init__(self, config: EyeConfig):
        self.config = config
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.focus_queue = queue.Queue()
        
        # Focus position
        self.focus_x = 0.5
        self.focus_y = 0.5
        self.focus_radius = 0.15
        
        # Motion detection
        self.prev_gray = None
        
        # Auto-calibration
        self.calibration_mode = True
        self.calibration_frames = 0
        self.motion_samples = deque(maxlen=300)
        self.area_samples = deque(maxlen=300)
        
        # Dynamic thresholds
        self.pixel_threshold = 20  # Initial pixel difference threshold
        self.area_threshold = 100  # Minimum contour area
        self.motion_stats = {}
        
        # Try to load calibration
        self.load_calibration()
        
    def process_frame(self, frame):
        """Process frame with motion tracking"""
        h, w = frame.shape[:2]
        
        # Apply corrections if needed
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.config.flip_vertical:
            frame = cv2.flip(frame, 0)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        thresh = None  # Initialize thresh
        
        if self.prev_gray is not None:
            # Motion detection
            diff = cv2.absdiff(gray, self.prev_gray)
            
            # Threshold to find motion
            _, thresh = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
            
            # Find motion contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area
            valid_contours = [c for c in contours if cv2.contourArea(c) > self.area_threshold]
            
            # Collect calibration data
            if self.calibration_mode and contours:
                for c in contours:
                    area = cv2.contourArea(c)
                    if area > 10:  # Minimal area
                        # Sample the average intensity in this region
                        mask = np.zeros_like(diff)
                        cv2.drawContours(mask, [c], -1, 255, -1)
                        mean_diff = cv2.mean(diff, mask=mask)[0]
                        self.motion_samples.append(mean_diff)
                        self.area_samples.append(area)
                
                self.calibration_frames += 1
                if self.calibration_frames % 30 == 0:
                    self._update_calibration()
            
            # Update focus based on largest motion
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                
                if M["m00"] > 0:
                    # Calculate center
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Normalize
                    new_x = cx / w
                    new_y = cy / h
                    
                    # Check if in periphery (outside focus circle)
                    dist_from_focus = np.sqrt((new_x - self.focus_x)**2 + (new_y - self.focus_y)**2)
                    
                    if dist_from_focus > self.focus_radius:
                        # Smooth movement
                        alpha = 0.3
                        self.focus_x = alpha * new_x + (1 - alpha) * self.focus_x
                        self.focus_y = alpha * new_y + (1 - alpha) * self.focus_y
                        
                        # Send focus update
                        area = cv2.contourArea(largest_contour)
                        confidence = min(1.0, area / (w * h * 0.1))  # Normalize by 10% of frame
                        
                        focus = FocusPoint(
                            x=self.focus_x,
                            y=self.focus_y,
                            confidence=confidence,
                            timestamp=time.time(),
                            source=f"{self.config.eye_id}_motion"
                        )
                        self.focus_queue.put(focus)
        
        self.prev_gray = gray
        
        # Visualize
        return self._visualize(frame, h, w, thresh)
    
    def _update_calibration(self):
        """Update thresholds based on collected data"""
        if len(self.motion_samples) < 50:
            return
            
        # Calculate statistics
        motion_array = np.array(self.motion_samples)
        area_array = np.array(self.area_samples)
        
        self.motion_stats = {
            'motion_mean': np.mean(motion_array),
            'motion_std': np.std(motion_array),
            'motion_p90': np.percentile(motion_array, 90),
            'motion_p95': np.percentile(motion_array, 95),
            'area_p50': np.percentile(area_array, 50),
            'area_p90': np.percentile(area_array, 90)
        }
        
        # Update thresholds
        # Pixel threshold: slightly below typical motion intensity
        self.pixel_threshold = max(10, int(self.motion_stats['motion_p90'] * 0.7))
        
        # Area threshold: filter out noise
        self.area_threshold = max(50, int(self.motion_stats['area_p50']))
        
        # Exit calibration after 10 seconds
        if self.calibration_frames > 300:
            self.calibration_mode = False
            print(f"{self.config.eye_id} eye calibration complete:")
            print(f"  Pixel threshold: {self.pixel_threshold}")
            print(f"  Area threshold: {self.area_threshold}")
            print(f"  Motion stats: {self.motion_stats}")
    
    def _visualize(self, frame, h, w, thresh):
        """Visualize with debug info"""
        result = frame.copy()
        
        # Show motion areas
        if thresh is not None:
            thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            thresh_color[:,:,0] = 0  # Remove blue
            thresh_color[:,:,1] = 0  # Remove green
            result = cv2.addWeighted(result, 0.7, thresh_color, 0.3, 0)
        
        # Draw focus circle
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        radius_px = int(self.focus_radius * min(h, w))
        
        color = (255, 100, 0) if self.config.eye_id == "left" else (0, 100, 255)
        cv2.circle(result, (focus_px, focus_py), radius_px, color, 3)
        
        # Crosshair
        cv2.line(result, (focus_px - 20, focus_py), (focus_px + 20, focus_py), color, 2)
        cv2.line(result, (focus_px, focus_py - 20), (focus_px, focus_py + 20), color, 2)
        
        # Labels
        cv2.putText(result, f"{self.config.eye_id.upper()} EYE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(result, f"Focus: ({self.focus_x:.2f}, {self.focus_y:.2f})", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calibration status
        if self.calibration_mode:
            cv2.putText(result, "CALIBRATING...", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            progress = min(100, (self.calibration_frames / 300) * 100)
            cv2.putText(result, f"Progress: {progress:.0f}%", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(result, f"Calibrated", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"Pixel: {self.pixel_threshold}, Area: {self.area_threshold}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return result
    
    def save_calibration(self, filename=None):
        """Save calibration data"""
        if filename is None:
            filename = f"{self.config.eye_id}_eye_calibration_v2.json"
            
        cal_data = {
            'eye_id': self.config.eye_id,
            'timestamp': time.time(),
            'pixel_threshold': self.pixel_threshold,
            'area_threshold': self.area_threshold,
            'motion_stats': self.motion_stats,
            'calibration_frames': self.calibration_frames
        }
        
        with open(filename, 'w') as f:
            json.dump(cal_data, f, indent=2)
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename=None):
        """Load calibration data"""
        if filename is None:
            filename = f"{self.config.eye_id}_eye_calibration_v2.json"
            
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                cal_data = json.load(f)
            
            self.pixel_threshold = cal_data['pixel_threshold']
            self.area_threshold = cal_data['area_threshold']
            self.motion_stats = cal_data.get('motion_stats', {})
            self.calibration_mode = False
            
            print(f"Calibration loaded from {filename}")
            return True
        return False
    
    def get_pipeline(self, width=640, height=480, fps=30):
        """GStreamer pipeline"""
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

class ImprovedBinocularConsciousness:
    """Binocular system with working motion tracking"""
    
    def __init__(self):
        # Configure eyes
        self.left_eye = AutoCalibratingEyeV2(EyeConfig(
            eye_id="left",
            sensor_id=0,
            position_offset=-1.5,
            rotation_offset=0.0
        ))
        
        self.right_eye = AutoCalibratingEyeV2(EyeConfig(
            eye_id="right", 
            sensor_id=1,
            position_offset=1.5,
            rotation_offset=0.0
        ))
        
        # Correlation engine
        self.correlator = StereoCorrelationEngine(baseline_inches=3.0)
        self.running = False
        
    def run(self):
        """Main loop"""
        self.running = True
        
        # Start threads
        left_thread = threading.Thread(target=self._eye_loop, args=(self.left_eye,))
        right_thread = threading.Thread(target=self._eye_loop, args=(self.right_eye,))
        correlation_thread = threading.Thread(target=self._correlation_loop)
        
        left_thread.start()
        right_thread.start()
        correlation_thread.start()
        
        print("\n🧠 Improved Binocular Vision with Auto-Calibration")
        print("="*50)
        print("Motion tracking working + auto-calibration!")
        print("\nPress 'q' to quit")
        print("Press 's' to save calibration")
        print("Press 'r' to reset calibration")
        
        while self.running:
            with self.left_eye.frame_lock:
                left_frame = self.left_eye.current_frame
            with self.right_eye.frame_lock:
                right_frame = self.right_eye.current_frame
                
            if left_frame is not None and right_frame is not None:
                stereo_view = np.hstack([left_frame, right_frame])
                
                # Add depth info if available
                if self.correlator.observation_history:
                    latest = self.correlator.observation_history[-1]
                    if latest.world_position:
                        depth = latest.world_position[2]
                        cv2.putText(stereo_view, f"Depth: {depth:.1f} inches", 
                                   (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2)
                
                cv2.imshow('Improved Binocular', stereo_view)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    self.left_eye.save_calibration()
                    self.right_eye.save_calibration()
                elif key == ord('r'):
                    self.left_eye.calibration_mode = True
                    self.right_eye.calibration_mode = True
                    self.left_eye.calibration_frames = 0
                    self.right_eye.calibration_frames = 0
                    print("Calibration reset")
                    
        # Cleanup
        left_thread.join()
        right_thread.join()
        correlation_thread.join()
        cv2.destroyAllWindows()
        
    def _eye_loop(self, eye):
        """Eye processing thread"""
        cap = cv2.VideoCapture(eye.get_pipeline(), cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print(f"Failed to open {eye.config.eye_id} eye")
            return
            
        print(f"{eye.config.eye_id} eye started")
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                vis_frame = eye.process_frame(frame)
                with eye.frame_lock:
                    eye.current_frame = vis_frame
                
        cap.release()
    
    def _correlation_loop(self):
        """Correlation thread"""
        left_focus = None
        right_focus = None
        
        while self.running:
            try:
                # Check for focus updates
                if not self.left_eye.focus_queue.empty():
                    left_focus = self.left_eye.focus_queue.get_nowait()
                    
                if not self.right_eye.focus_queue.empty():
                    right_focus = self.right_eye.focus_queue.get_nowait()
                    
                # Correlate if we have both
                if left_focus and right_focus:
                    time_diff = abs(left_focus.timestamp - right_focus.timestamp)
                    if time_diff < 0.1:  # Within 100ms
                        self.correlator.correlate(left_focus, right_focus)
                        
            except queue.Empty:
                pass
                
            time.sleep(0.01)

def main():
    consciousness = ImprovedBinocularConsciousness()
    
    try:
        consciousness.run()
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()