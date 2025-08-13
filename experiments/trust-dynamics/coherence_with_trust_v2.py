#!/usr/bin/env python3
"""
Coherence Engine with Improved Trust Dynamics
- Edge detection for camera meaningfulness
- Proper context state transitions
- Better sensor conflict detection
August 12, 2025
"""

import cv2
import numpy as np
import time
import signal
import sys
import threading
from collections import deque
from enum import Enum, auto

# Import the existing IMU module
sys.path.append('/home/sprout/ai-workspace/private-context/ai-dna-discovery/imu')
from yahboom_cmp10a import YahboomCMP10A

# Import logging effector
sys.path.append('/home/sprout/ai-workspace/private-context/ai-dna-discovery/coherence-engine/plugins/common')
from logging_effector import LoggingEffector

# Context states
class ContextState(Enum):
    STABLE = auto()
    MOVING = auto()
    UNSTABLE = auto()
    NOVEL = auto()

class ImprovedCoherence:
    def __init__(self):
        # Camera setup
        self.cap_l = None
        self.cap_r = None
        
        # IMU setup
        self.imu = None
        self.imu_thread = None
        self.use_real_imu = False
        
        # Coherence state
        self.reality_field = 0.5
        self.context_state = ContextState.STABLE
        
        # IMPROVED: Separate trust for each camera
        self.trust_weights = {
            "camera_left": 1.0,
            "camera_right": 1.0,
            "imu": 1.0
        }
        self.relevance_weights = {
            "camera_left": 1.0, 
            "camera_right": 1.0,
            "imu": 1.0
        }
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.tick_count = 0
        self.start_time = time.time()
        self.last_time = time.time()
        
        # IMPROVED: Better sensor metrics
        self.camera_left_quality = 1.0   # Edge-based quality
        self.camera_right_quality = 1.0
        self.camera_motion = 0.0
        self.imu_stability = 0.9
        self.imu_motion_magnitude = 0.0
        
        # Context tracking for better state transitions
        self.motion_history = deque(maxlen=30)  # 1 second at 30Hz
        self.stability_history = deque(maxlen=30)
        
        self.field_history = deque(maxlen=100)
        self.imu_data = {
            "acceleration": [0.0, 0.0, 9.81],
            "gyroscope": [0.0, 0.0, 0.0],
            "magnetometer": [30.0, -10.0, 45.0],
            "orientation": [0.0, 0.0, 0.0],
            "temperature": 25.0
        }
        
        # System control
        self.running = True
        
        # Logging effector
        self.logger = LoggingEffector(
            log_dir="experiments/trust-dynamics",
            log_rate=10.0
        )
        
    def gst_pipeline(self, sensor_id=0):
        """Create GStreamer pipeline for CSI camera"""
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
            f"video/x-raw(memory:NVMM), width=1920, height=1080, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        
    def assess_camera_quality(self, frame):
        """
        Assess if camera data is meaningful using edge detection.
        Returns quality score 0-1 based on edge presence and distribution.
        """
        if frame is None:
            return 0.0
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge detection using Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 2. Calculate edge metrics
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        
        # 3. Check for meaningful structure
        # Low edge mean = likely covered/black
        # Very low std = uniform image (covered or pointing at blank wall)
        structure_score = min(edge_mean / 50.0, 1.0)  # Normalize
        variation_score = min(edge_std / 30.0, 1.0)
        
        # 4. Check brightness (too dark or too bright = unreliable)
        brightness = np.mean(gray)
        brightness_score = 1.0
        if brightness < 20:  # Too dark (likely covered)
            brightness_score = brightness / 20.0
        elif brightness > 235:  # Too bright (saturated)
            brightness_score = (255 - brightness) / 20.0
            
        # 5. Combine metrics
        quality = structure_score * 0.4 + variation_score * 0.4 + brightness_score * 0.2
        
        return min(max(quality, 0.0), 1.0)
        
    def compute_camera_motion(self, frame_l, frame_r):
        """Compute motion from camera frames with quality weighting"""
        if frame_l is None and frame_r is None:
            return 0.0
            
        motion_l = 0.0
        motion_r = 0.0
        
        # Left camera motion (if quality is good)
        if frame_l is not None and self.camera_left_quality > 0.3:
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            lap_l = cv2.Laplacian(gray_l, cv2.CV_64F).var()
            motion_l = min(lap_l / 1000.0, 1.0) * self.camera_left_quality
            
        # Right camera motion (if quality is good)  
        if frame_r is not None and self.camera_right_quality > 0.3:
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            lap_r = cv2.Laplacian(gray_r, cv2.CV_64F).var()
            motion_r = min(lap_r / 1000.0, 1.0) * self.camera_right_quality
            
        # Weighted average based on quality
        total_quality = self.camera_left_quality + self.camera_right_quality
        if total_quality > 0:
            return (motion_l * self.camera_left_quality + 
                   motion_r * self.camera_right_quality) / total_quality
        return 0.0
        
    def update_context(self):
        """IMPROVED: Better context state detection"""
        # Add current readings to history
        self.motion_history.append(self.camera_motion)
        self.stability_history.append(self.imu_stability)
        
        # Calculate averages
        avg_motion = np.mean(self.motion_history) if self.motion_history else 0
        avg_stability = np.mean(self.stability_history) if self.stability_history else 1
        
        # Determine context based on sensor agreement and activity
        if avg_motion < 0.1 and avg_stability > 0.9 and self.imu_motion_magnitude < 0.1:
            # Low motion, high stability, low gyro = STABLE
            self.context_state = ContextState.STABLE
        elif avg_motion > 0.3 or self.imu_motion_magnitude > 0.5:
            # Clear motion detected = MOVING
            self.context_state = ContextState.MOVING
        elif avg_stability < 0.5 or (self.camera_left_quality < 0.3 and self.camera_right_quality < 0.3):
            # Poor stability or poor camera quality = UNSTABLE
            self.context_state = ContextState.UNSTABLE
        elif len(self.field_history) > 10:
            recent_std = np.std(list(self.field_history)[-10:])
            if recent_std > 0.3:
                self.context_state = ContextState.NOVEL
            # else stays in current state
                
    def update_trust(self):
        """IMPROVED: Trust based on sensor quality and agreement"""
        # Update camera trust based on quality
        quality_threshold = 0.3
        trust_decay = 0.95
        trust_recovery = 1.02
        
        # Left camera trust
        if self.camera_left_quality < quality_threshold:
            self.trust_weights["camera_left"] *= trust_decay
        else:
            self.trust_weights["camera_left"] = min(1.0, 
                self.trust_weights["camera_left"] * trust_recovery)
            
        # Right camera trust  
        if self.camera_right_quality < quality_threshold:
            self.trust_weights["camera_right"] *= trust_decay
        else:
            self.trust_weights["camera_right"] = min(1.0,
                self.trust_weights["camera_right"] * trust_recovery)
            
        # IMU trust based on reasonable values
        gyro_mag = np.linalg.norm(self.imu_data["gyroscope"])
        if gyro_mag > 100:  # Unreasonably high gyro
            self.trust_weights["imu"] *= trust_decay
        else:
            self.trust_weights["imu"] = min(1.0,
                self.trust_weights["imu"] * trust_recovery)
            
        # Bonus/penalty for sensor agreement
        camera_avg_quality = (self.camera_left_quality + self.camera_right_quality) / 2
        
        # If cameras see motion but IMU is stable (or vice versa)
        if abs(self.camera_motion - (1.0 - self.imu_stability)) > 0.5:
            # Disagreement - reduce trust in both
            for sensor in ["camera_left", "camera_right", "imu"]:
                self.trust_weights[sensor] *= 0.98
        else:
            # Agreement - slight trust increase
            for sensor in ["camera_left", "camera_right", "imu"]:
                self.trust_weights[sensor] = min(1.0, 
                    self.trust_weights[sensor] * 1.001)
                    
    def update_weights(self):
        """Update relevance weights based on context"""
        if self.context_state == ContextState.STABLE:
            # In stable state, cameras more relevant for detecting changes
            self.relevance_weights = {
                "camera_left": 1.0,
                "camera_right": 1.0,
                "imu": 0.5
            }
        elif self.context_state == ContextState.MOVING:
            # When moving, all sensors equally relevant
            self.relevance_weights = {
                "camera_left": 0.8,
                "camera_right": 0.8,
                "imu": 1.0
            }
        elif self.context_state == ContextState.UNSTABLE:
            # Unstable - rely more on whatever is working
            self.relevance_weights = {
                "camera_left": self.camera_left_quality,
                "camera_right": self.camera_right_quality,
                "imu": 1.0
            }
        else:  # NOVEL
            # Novel - heighten all sensors
            self.relevance_weights = {
                "camera_left": 1.0,
                "camera_right": 1.0,
                "imu": 1.0
            }
            
    def compute_reality_field(self):
        """Compute reality field from weighted sensors"""
        # Camera contribution (average of left and right)
        camera_l_contrib = (self.camera_motion * self.camera_left_quality *
                           self.trust_weights["camera_left"] * 
                           self.relevance_weights["camera_left"])
        camera_r_contrib = (self.camera_motion * self.camera_right_quality *
                           self.trust_weights["camera_right"] * 
                           self.relevance_weights["camera_right"])
        
        # IMU contribution
        imu_contrib = (self.imu_stability * 
                      self.trust_weights["imu"] * 
                      self.relevance_weights["imu"])
                      
        # Normalize
        total_weight = (self.trust_weights["camera_left"] * self.relevance_weights["camera_left"] +
                       self.trust_weights["camera_right"] * self.relevance_weights["camera_right"] +
                       self.trust_weights["imu"] * self.relevance_weights["imu"])
                       
        if total_weight > 0:
            self.reality_field = (camera_l_contrib + camera_r_contrib + imu_contrib) / total_weight
        else:
            self.reality_field = 0.5
            
        # Clamp to valid range
        self.reality_field = max(0.0, min(1.0, self.reality_field))
        
        # Add to history
        self.field_history.append(self.reality_field)
        
    def run(self):
        """Main loop with phase management"""
        # Initialize cameras and IMU
        self.initialize()
        
        # Phase definitions
        phases = [
            ("baseline", 30, "Normal operation"),
            ("left_occlusion", 30, "Cover LEFT camera"),
            ("motion_conflict", 30, "Uncover, then shake device"),
            ("recovery", 30, "Return to normal"),
            ("full_occlusion", 30, "Cover BOTH cameras")
        ]
        
        phase_idx = 0
        phase_start = time.time()
        self.logger.set_phase(phases[0][0])
        
        print(f"\n{'='*50}")
        print(f"PHASE: {phases[0][0].upper()} ({phases[0][1]}s)")
        print(f"Action: {phases[0][2]}")
        print(f"{'='*50}\n")
        
        while self.running:
            # Check phase transition
            if phase_idx < len(phases) - 1 and time.time() - phase_start > phases[phase_idx][1]:
                phase_idx += 1
                phase_start = time.time()
                self.logger.set_phase(phases[phase_idx][0])
                
                print(f"\n{'='*50}")
                print(f"PHASE: {phases[phase_idx][0].upper()} ({phases[phase_idx][1]}s)")
                print(f"Action: {phases[phase_idx][2]}")
                print(f"{'='*50}\n")
                
            # Read camera frames
            ret_l, frame_l = self.cap_l.read()
            ret_r, frame_r = self.cap_r.read()
            
            # Assess camera quality (meaningfulness)
            self.camera_left_quality = self.assess_camera_quality(frame_l) if ret_l else 0.0
            self.camera_right_quality = self.assess_camera_quality(frame_r) if ret_r else 0.0
            
            # Compute motion
            self.camera_motion = self.compute_camera_motion(frame_l, frame_r)
            
            # Update IMU
            if self.imu_data:
                gyro_mag = np.linalg.norm(self.imu_data["gyroscope"])
                self.imu_motion_magnitude = gyro_mag / 100.0  # Normalize
                self.imu_stability = 1.0 / (1.0 + gyro_mag * 10)
            
            # Update coherence
            self.update_context()
            self.update_weights()
            self.update_trust()
            self.compute_reality_field()
            
            # Log data
            context = {
                "tick": self.tick_count,
                "state": self.context_state.name,
                "trust_weights": self.trust_weights.copy(),
                "camera_motion": self.camera_motion,
                "camera_left_quality": self.camera_left_quality,
                "camera_right_quality": self.camera_right_quality,
                "imu_stability": self.imu_stability,
                "imu_motion_magnitude": self.imu_motion_magnitude,
                "imu_data": self.imu_data.copy()
            }
            self.logger.effect(self.reality_field, context)
            
            # Display
            if frame_l is not None or frame_r is not None:
                dashboard = self.create_dashboard(frame_l, frame_r)
                cv2.imshow("Trust Dynamics V2", dashboard)
                
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            # Update tick
            self.tick_count += 1
            
            # Status
            if self.tick_count % 30 == 0:
                print(f"Tick {self.tick_count} | "
                      f"RF: {self.reality_field:.2f} | "
                      f"Context: {self.context_state.name} | "
                      f"Trust: L={self.trust_weights['camera_left']:.2f} "
                      f"R={self.trust_weights['camera_right']:.2f} "
                      f"I={self.trust_weights['imu']:.2f} | "
                      f"Quality: L={self.camera_left_quality:.2f} "
                      f"R={self.camera_right_quality:.2f}")
                      
        self.shutdown()
        
    def initialize(self):
        """Initialize cameras and IMU"""
        print("\nInitializing improved coherence engine...")
        
        # Initialize cameras
        self.cap_l = cv2.VideoCapture(self.gst_pipeline(0), cv2.CAP_GSTREAMER)
        self.cap_r = cv2.VideoCapture(self.gst_pipeline(1), cv2.CAP_GSTREAMER)
        
        # Initialize IMU
        try:
            self.imu = YahboomCMP10A(port="/dev/ttyUSB0", baud=9600)
            self.use_real_imu = True
            print("✓ IMU connected")
        except:
            print("✗ Using simulated IMU")
            
        print("✓ System ready\n")
        
    def create_dashboard(self, frame_l, frame_r):
        """Create visualization dashboard"""
        dashboard = np.zeros((600, 1200, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)
        
        # Camera feeds (smaller)
        if frame_l is not None:
            small_l = cv2.resize(frame_l, (400, 225))
            dashboard[20:245, 20:420] = small_l
            
        if frame_r is not None:
            small_r = cv2.resize(frame_r, (400, 225))
            dashboard[20:245, 440:840] = small_r
            
        # Quality indicators
        cv2.putText(dashboard, f"L Quality: {self.camera_left_quality:.2f}",
                   (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(dashboard, f"R Quality: {self.camera_right_quality:.2f}",
                   (450, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                   
        # Reality field visualization
        center_x, center_y = 600, 400
        radius = int(50 * (1 + self.reality_field))
        color_intensity = int(255 * self.reality_field)
        color = (0, color_intensity, 255 - color_intensity)
        cv2.circle(dashboard, (center_x, center_y), radius, color, -1)
        
        # Status text
        cv2.putText(dashboard, f"Reality Field: {self.reality_field:.3f}",
                   (500, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(dashboard, f"Context: {self.context_state.name}",
                   (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
                   
        # Trust weights
        y = 350
        for sensor, trust in self.trust_weights.items():
            cv2.putText(dashboard, f"{sensor}: {trust:.2f}",
                       (900, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 25
            
        return dashboard
        
    def shutdown(self):
        """Clean shutdown"""
        print("\nShutting down...")
        
        # Finalize logging
        stats = self.logger.finalize()
        analysis = self.logger.analyze()
        
        print(f"\nExperiment complete!")
        print(f"Logged {stats['entries_logged']} entries")
        print(f"Trust changes: {analysis.get('trust_changes', {})}")
        
        # Close cameras
        if self.cap_l:
            self.cap_l.release()
        if self.cap_r:
            self.cap_r.release()
            
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    experiment = ImprovedCoherence()
    try:
        experiment.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        experiment.shutdown()