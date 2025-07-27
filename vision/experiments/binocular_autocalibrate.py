#!/usr/bin/env python3
"""
Binocular Vision with Auto-Calibration
Automatically adjusts motion detection thresholds based on environment
"""

import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import json
import os

from binocular_consciousness import (
    EyeConfig, FocusPoint, StereoObservation, 
    CognitionHook, IndependentEye, StereoCorrelationEngine
)

class AutoCalibratingEye(IndependentEye):
    """Eye with automatic threshold calibration"""
    
    def __init__(self, config: EyeConfig):
        super().__init__(config)
        
        # Calibration state
        self.calibration_mode = True
        self.calibration_frames = 0
        self.calibration_samples = deque(maxlen=300)  # 10 seconds at 30fps
        
        # Dynamic thresholds - start with reasonable defaults
        self.noise_floor = 0.001
        self.motion_threshold = 0.01  # Lower initial threshold
        self.significant_motion = 0.03  # Lower for better response
        
        # Make sure focus starts centered
        self.focus_x = 0.5
        self.focus_y = 0.5
        
        # Motion statistics
        self.motion_stats = {
            'mean': 0.0,
            'std': 0.0,
            'percentile_50': 0.0,
            'percentile_90': 0.0,
            'percentile_95': 0.0,
            'percentile_99': 0.0
        }
        
    def process_frame(self, frame):
        """Process frame with auto-calibration"""
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
            
            # Collect calibration samples
            if self.calibration_mode:
                self._collect_calibration_data(diff)
            
            # Update motion heatmap with dynamic thresholds
            self._update_motion_heatmap(diff, h, w)
            
            # Find peaks with calibrated thresholds
            peaks = self._find_motion_peaks()
            
            # Update focus based on motion (even during calibration with defaults)
            if peaks:
                # Use current threshold or default during calibration
                threshold = self.significant_motion if not self.calibration_mode else 0.02
                
                if peaks[0]['strength'] > threshold:
                    new_focus = FocusPoint(
                        x=peaks[0]['x'],
                        y=peaks[0]['y'],
                        confidence=min(1.0, peaks[0]['strength'] / max(threshold, 0.001)),
                        timestamp=time.time(),
                        source=f"{self.config.eye_id}_motion"
                    )
                    
                    # Smooth focus updates
                    alpha = 0.3  # Smoothing factor
                    self.focus_x = alpha * new_focus.x + (1 - alpha) * self.focus_x
                    self.focus_y = alpha * new_focus.y + (1 - alpha) * self.focus_y
                    
                    # Send significant updates
                    if new_focus.confidence > 0.3:  # Lower threshold during calibration
                        self.focus_queue.put(new_focus)
        
        self.prev_gray = gray
        
        # Update curiosity
        self.curiosity *= 0.95
        self.curiosity = max(0.2, self.curiosity)
            
        return self._visualize_with_calibration(frame, h, w)
    
    def _collect_calibration_data(self, diff):
        """Collect motion samples for calibration"""
        # Sample motion values from different regions
        h, w = diff.shape
        grid_size = 16  # Finer grid for calibration
        
        samples = []
        for i in range(0, h, h // grid_size):
            for j in range(0, w, w // grid_size):
                region = diff[i:i + h // grid_size, j:j + w // grid_size]
                if region.size > 0:
                    samples.append(np.mean(region) / 255.0)
        
        self.calibration_samples.extend(samples)
        self.calibration_frames += 1
        
        # Update calibration every 30 frames (1 second)
        if self.calibration_frames % 30 == 0:
            self._update_calibration()
    
    def _update_calibration(self):
        """Update motion thresholds based on collected data"""
        if len(self.calibration_samples) < 100:
            return
            
        samples = np.array(self.calibration_samples)
        
        # Calculate statistics
        self.motion_stats['mean'] = np.mean(samples)
        self.motion_stats['std'] = np.std(samples)
        self.motion_stats['percentile_50'] = np.percentile(samples, 50)
        self.motion_stats['percentile_90'] = np.percentile(samples, 90)
        self.motion_stats['percentile_95'] = np.percentile(samples, 95)
        self.motion_stats['percentile_99'] = np.percentile(samples, 99)
        
        # Set thresholds based on statistics
        self.noise_floor = self.motion_stats['percentile_50']
        self.ambient_motion = self.motion_stats['percentile_90']
        self.motion_threshold = self.ambient_motion * 1.5
        self.significant_motion = self.motion_stats['percentile_95']
        
        # Exit calibration mode after sufficient data
        if self.calibration_frames > 300:  # 10 seconds
            self.calibration_mode = False
            print(f"{self.config.eye_id} eye calibration complete:")
            print(f"  Noise floor: {self.noise_floor:.4f}")
            print(f"  Ambient motion: {self.ambient_motion:.4f}")
            print(f"  Motion threshold: {self.motion_threshold:.4f}")
            print(f"  Significant motion: {self.significant_motion:.4f}")
    
    def _find_motion_peaks(self):
        """Find areas of peak motion with calibrated thresholds"""
        peaks = []
        
        # Use current threshold or reasonable default during calibration
        threshold = self.motion_threshold if not self.calibration_mode else 0.005
        ambient = self.ambient_motion if self.ambient_motion > 0 else 0.01
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                motion_val = self.motion_heatmap[i, j]
                
                # Only consider motion above threshold
                if motion_val > threshold:
                    # Calculate significance ratio
                    if self.calibration_mode:
                        # Simple significance during calibration
                        significance = min(1.0, motion_val / 0.05)
                    else:
                        significance = (motion_val - ambient) / (self.significant_motion - ambient)
                        significance = max(0, min(1, significance))  # Clamp to [0, 1]
                    
                    peaks.append({
                        'x': (j + 0.5) / self.motion_grid_size,
                        'y': (i + 0.5) / self.motion_grid_size,
                        'strength': motion_val,
                        'significance': significance,
                        'pa_ratio': motion_val / ambient if ambient > 0 else 1.0
                    })
        
        return sorted(peaks, key=lambda p: p['significance'], reverse=True)
    
    def _visualize_with_calibration(self, frame, h, w):
        """Visualization including calibration info"""
        result = self._visualize(frame, h, w)
        
        # Add calibration status
        if self.calibration_mode:
            cv2.putText(result, "CALIBRATING...", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            progress = min(100, (self.calibration_frames / 300) * 100)
            cv2.putText(result, f"Progress: {progress:.0f}%", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(result, "Calibrated", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show thresholds
        cv2.putText(result, f"Ambient: {self.ambient_motion:.4f}", (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(result, f"Threshold: {self.motion_threshold:.4f}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return result
    
    def save_calibration(self, filename=None):
        """Save calibration data"""
        if filename is None:
            filename = f"{self.config.eye_id}_eye_calibration.json"
            
        cal_data = {
            'eye_id': self.config.eye_id,
            'timestamp': time.time(),
            'noise_floor': self.noise_floor,
            'ambient_motion': self.ambient_motion,
            'motion_threshold': self.motion_threshold,
            'significant_motion': self.significant_motion,
            'motion_stats': self.motion_stats,
            'calibration_frames': self.calibration_frames
        }
        
        with open(filename, 'w') as f:
            json.dump(cal_data, f, indent=2)
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename=None):
        """Load calibration data"""
        if filename is None:
            filename = f"{self.config.eye_id}_eye_calibration.json"
            
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                cal_data = json.load(f)
            
            self.noise_floor = cal_data['noise_floor']
            self.ambient_motion = cal_data['ambient_motion']
            self.motion_threshold = cal_data['motion_threshold']
            self.significant_motion = cal_data['significant_motion']
            self.motion_stats = cal_data['motion_stats']
            self.calibration_mode = False
            
            print(f"Calibration loaded from {filename}")
            return True
        return False

class AdaptiveBinocularConsciousness:
    """Binocular system with auto-calibrating eyes"""
    
    def __init__(self):
        # Configure auto-calibrating eyes
        self.left_eye = AutoCalibratingEye(EyeConfig(
            eye_id="left",
            sensor_id=0,
            position_offset=-1.5,
            rotation_offset=0.0
        ))
        
        self.right_eye = AutoCalibratingEye(EyeConfig(
            eye_id="right", 
            sensor_id=1,
            position_offset=1.5,
            rotation_offset=0.0
        ))
        
        # Try to load existing calibration
        self.left_eye.load_calibration()
        self.right_eye.load_calibration()
        
        # Correlation engine
        self.correlator = StereoCorrelationEngine(baseline_inches=3.0)
        
        self.running = False
        
    def run(self):
        """Main processing loop"""
        self.running = True
        
        # Start capture threads
        left_thread = threading.Thread(target=self._eye_loop, args=(self.left_eye,))
        right_thread = threading.Thread(target=self._eye_loop, args=(self.right_eye,))
        
        left_thread.start()
        right_thread.start()
        
        print("\n🧠 Auto-Calibrating Binocular Consciousness")
        print("="*50)
        print("The system will calibrate to your environment automatically")
        print("Move around normally for 10 seconds during calibration")
        print("\nPress 'q' to quit")
        print("Press 's' to save calibration")
        print("Press 'r' to reset calibration")
        
        while self.running:
            # Get current frames
            with self.left_eye.frame_lock:
                left_frame = self.left_eye.current_frame
            with self.right_eye.frame_lock:
                right_frame = self.right_eye.current_frame
                
            if left_frame is not None and right_frame is not None:
                # Create visualization
                stereo_view = np.hstack([left_frame, right_frame])
                
                # Add title
                status = "CALIBRATING" if (self.left_eye.calibration_mode or self.right_eye.calibration_mode) else "CALIBRATED"
                color = (0, 255, 255) if status == "CALIBRATING" else (0, 255, 0)
                cv2.putText(stereo_view, f"Auto-Calibrating Binocular Vision - {status}", 
                           (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow('Adaptive Binocular', stereo_view)
                
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
        cv2.destroyAllWindows()
        
    def _eye_loop(self, eye):
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

def main():
    consciousness = AdaptiveBinocularConsciousness()
    
    try:
        consciousness.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()