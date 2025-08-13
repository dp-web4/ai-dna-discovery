#!/usr/bin/env python3
"""
Coherence Engine with Improved Trust Dynamics V3
- Adjusted edge detection thresholds
- IMU reading thread
- Better context state detection
- Canny edge detection for occlusion
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

class CoherenceV3:
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
        
        # Separate trust for each camera
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
        
        # Sensor metrics
        self.camera_left_quality = 1.0
        self.camera_right_quality = 1.0
        self.camera_motion = 0.0
        self.imu_stability = 0.9
        self.imu_motion_magnitude = 0.0
        
        # Previous frames for motion detection
        self.prev_frame_l = None
        self.prev_frame_r = None
        
        # Context tracking for better state transitions
        self.motion_history = deque(maxlen=30)
        self.stability_history = deque(maxlen=30)
        self.quality_history = deque(maxlen=30)
        
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
        V3: Better occlusion detection using Canny edges and adjusted thresholds
        """
        if frame is None:
            return 0.0
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Canny edge detection for better occlusion detection
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_ratio = edge_pixels / total_pixels
        
        # 2. Brightness check
        brightness = np.mean(gray)
        
        # 3. Contrast check (standard deviation)
        contrast = np.std(gray)
        
        # Debug info (every 30 frames)
        if self.tick_count % 30 == 0:
            print(f"    Edge ratio: {edge_ratio:.3f}, Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
        
        # Quality scoring
        # Edge ratio: 0.01-0.10 is typical for normal scenes
        # < 0.005 likely covered, > 0.15 likely noisy
        edge_score = 1.0
        if edge_ratio < 0.005:  # Too few edges - likely covered
            edge_score = edge_ratio / 0.005
        elif edge_ratio > 0.15:  # Too many edges - likely noise
            edge_score = max(0, 1.0 - (edge_ratio - 0.15) / 0.1)
        
        # Brightness: 10-245 is acceptable range
        brightness_score = 1.0
        if brightness < 10:  # Too dark - likely covered
            brightness_score = brightness / 10.0
        elif brightness > 245:  # Too bright - saturated
            brightness_score = (255 - brightness) / 10.0
            
        # Contrast: < 5 is likely uniform (covered)
        contrast_score = min(contrast / 20.0, 1.0)
        
        # Combine metrics
        quality = edge_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2
        
        return max(0.0, min(1.0, quality))
        
    def compute_camera_motion(self, frame_l, frame_r):
        """
        V3: Better motion detection using frame differencing
        """
        motion_l = 0.0
        motion_r = 0.0
        
        # Left camera motion
        if frame_l is not None and self.prev_frame_l is not None and self.camera_left_quality > 0.3:
            gray_curr = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame_l, cv2.COLOR_BGR2GRAY)
            
            # Frame difference
            diff = cv2.absdiff(gray_curr, gray_prev)
            motion_pixels = np.sum(diff > 30)  # Threshold for motion
            total_pixels = diff.shape[0] * diff.shape[1]
            motion_l = min(motion_pixels / (total_pixels * 0.1), 1.0)  # Normalize
            
        self.prev_frame_l = frame_l
            
        # Right camera motion
        if frame_r is not None and self.prev_frame_r is not None and self.camera_right_quality > 0.3:
            gray_curr = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame_r, cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray_curr, gray_prev)
            motion_pixels = np.sum(diff > 30)
            total_pixels = diff.shape[0] * diff.shape[1]
            motion_r = min(motion_pixels / (total_pixels * 0.1), 1.0)
            
        self.prev_frame_r = frame_r
        
        # Weighted average based on quality
        total_quality = self.camera_left_quality + self.camera_right_quality
        if total_quality > 0:
            return (motion_l * self.camera_left_quality + 
                   motion_r * self.camera_right_quality) / total_quality
        return 0.0
        
    def read_imu_loop(self):
        """Background thread to read real IMU data"""
        while self.running and self.imu:
            try:
                # Read available data into buffer
                if self.imu.ser.in_waiting:
                    data = self.imu.ser.read(self.imu.ser.in_waiting)
                    if data:
                        self.imu.buffer += data
                        
                # Process buffer for complete packets
                while len(self.imu.buffer) >= 11:
                    # Look for packet header (0x55)
                    start_idx = self.imu.buffer.find(b'\x55')
                    if start_idx == -1:
                        self.imu.buffer = b''
                        break
                        
                    # Check if we have a complete packet
                    if len(self.imu.buffer) >= start_idx + 11:
                        packet = self.imu.buffer[start_idx:start_idx + 11]
                        self.imu.buffer = self.imu.buffer[start_idx + 11:]
                        
                        # Parse the packet
                        parsed = self.imu.parse_packet(packet)
                        if parsed:
                            # Update our IMU data based on packet type
                            if parsed['type'] == 'accel':
                                self.imu_data["acceleration"] = [
                                    parsed['x'], parsed['y'], parsed['z']
                                ]
                                self.imu_data["temperature"] = parsed['temp']
                            elif parsed['type'] == 'gyro':
                                self.imu_data["gyroscope"] = [
                                    parsed['x'], parsed['y'], parsed['z']
                                ]
                            elif parsed['type'] == 'angle':
                                self.imu_data["orientation"] = [
                                    parsed['roll'], parsed['pitch'], parsed['yaw']
                                ]
                            elif parsed['type'] == 'mag':
                                self.imu_data["magnetometer"] = [
                                    parsed['x'], parsed['y'], parsed['z']
                                ]
                    else:
                        break
                        
            except Exception as e:
                if self.tick_count % 100 == 0:  # Only print occasionally
                    print(f"IMU read error: {e}")
                    
            time.sleep(0.001)  # 1000Hz potential read rate
        
    def update_context(self):
        """V3: Better context state detection with proper thresholds"""
        # Add current readings to history
        self.motion_history.append(self.camera_motion)
        self.stability_history.append(self.imu_stability)
        avg_quality = (self.camera_left_quality + self.camera_right_quality) / 2
        self.quality_history.append(avg_quality)
        
        # Calculate averages
        avg_motion = np.mean(self.motion_history) if self.motion_history else 0
        avg_stability = np.mean(self.stability_history) if self.stability_history else 1
        avg_quality = np.mean(self.quality_history) if self.quality_history else 1
        
        # Debug context decision (occasionally)
        if self.tick_count % 60 == 0:
            print(f"    Context: motion={avg_motion:.2f}, stability={avg_stability:.2f}, "
                  f"imu_mag={self.imu_motion_magnitude:.2f}, quality={avg_quality:.2f}")
        
        # Determine context based on sensor agreement and activity
        if avg_motion < 0.05 and avg_stability > 0.95 and self.imu_motion_magnitude < 0.01:
            # Very low motion, very high stability = STABLE
            self.context_state = ContextState.STABLE
        elif avg_motion > 0.1 or self.imu_motion_magnitude > 0.05:
            # Clear motion detected = MOVING
            self.context_state = ContextState.MOVING
        elif avg_quality < 0.5:
            # Poor sensor quality = UNSTABLE
            self.context_state = ContextState.UNSTABLE
        elif len(self.field_history) > 10:
            recent_std = np.std(list(self.field_history)[-10:])
            if recent_std > 0.3:
                self.context_state = ContextState.NOVEL
            # else stays in current state
                
    def update_trust(self):
        """V3: Trust based on sensor quality and agreement"""
        quality_threshold = 0.5  # More reasonable threshold
        trust_decay = 0.97  # Slower decay
        trust_recovery = 1.01  # Slower recovery
        
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
        if gyro_mag > 500:  # Very high gyro (degrees/sec)
            self.trust_weights["imu"] *= trust_decay
        else:
            self.trust_weights["imu"] = min(1.0,
                self.trust_weights["imu"] * trust_recovery)
            
        # Ensure minimum trust (never goes to absolute zero)
        for sensor in self.trust_weights:
            if self.trust_weights[sensor] < 0.01:
                self.trust_weights[sensor] = 0.01
                
    def update_weights(self):
        """Update relevance weights based on context"""
        if self.context_state == ContextState.STABLE:
            self.relevance_weights = {
                "camera_left": 1.0,
                "camera_right": 1.0,
                "imu": 0.5
            }
        elif self.context_state == ContextState.MOVING:
            self.relevance_weights = {
                "camera_left": 0.8,
                "camera_right": 0.8,
                "imu": 1.0
            }
        elif self.context_state == ContextState.UNSTABLE:
            # Rely on whatever has quality
            self.relevance_weights = {
                "camera_left": max(0.1, self.camera_left_quality),
                "camera_right": max(0.1, self.camera_right_quality),
                "imu": 1.0
            }
        else:  # NOVEL
            self.relevance_weights = {
                "camera_left": 1.0,
                "camera_right": 1.0,
                "imu": 1.0
            }
            
    def compute_reality_field(self):
        """Compute reality field from weighted sensors"""
        # Camera contribution
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
            
        self.reality_field = max(0.0, min(1.0, self.reality_field))
        self.field_history.append(self.reality_field)
        
    def run(self):
        """Main loop with phase management"""
        # Initialize cameras and IMU
        self.initialize()
        
        # Phase definitions
        phases = [
            ("baseline", 30, "Normal operation - establish baseline"),
            ("left_occlusion", 30, ">>> COVER LEFT CAMERA NOW <<<"),
            ("motion_conflict", 30, ">>> UNCOVER CAMERA, SHAKE DEVICE <<<"),
            ("recovery", 30, ">>> STOP SHAKING, RETURN TO NORMAL <<<"),
            ("full_occlusion", 30, ">>> COVER BOTH CAMERAS NOW <<<")
        ]
        
        phase_idx = 0
        phase_start = time.time()
        self.logger.set_phase(phases[0][0])
        
        print(f"\n{'='*60}")
        print(f"PHASE: {phases[0][0].upper()} ({phases[0][1]}s)")
        print(f"{phases[0][2]}")
        print(f"{'='*60}\n")
        
        while self.running:
            # Check phase transition
            if phase_idx < len(phases) - 1 and time.time() - phase_start > phases[phase_idx][1]:
                phase_idx += 1
                phase_start = time.time()
                self.logger.set_phase(phases[phase_idx][0])
                
                print(f"\n{'='*60}")
                print(f"PHASE: {phases[phase_idx][0].upper()} ({phases[phase_idx][1]}s)")
                print(f"{phases[phase_idx][2]}")
                print(f"{'='*60}\n")
                
            # Read camera frames
            ret_l, frame_l = self.cap_l.read()
            ret_r, frame_r = self.cap_r.read()
            
            # Assess camera quality
            self.camera_left_quality = self.assess_camera_quality(frame_l) if ret_l else 0.0
            self.camera_right_quality = self.assess_camera_quality(frame_r) if ret_r else 0.0
            
            # Compute motion
            self.camera_motion = self.compute_camera_motion(frame_l, frame_r)
            
            # Update IMU metrics
            if self.use_real_imu and self.imu_data:
                gyro_mag = np.linalg.norm(self.imu_data["gyroscope"])
                self.imu_motion_magnitude = min(gyro_mag / 200.0, 1.0)  # Normalize (200 deg/s = high motion)
                self.imu_stability = 1.0 / (1.0 + gyro_mag * 0.01)  # Adjusted scaling
            else:
                # Simulated IMU
                t = time.time()
                self.imu_data["gyroscope"] = [
                    10 * np.sin(t * 0.5),
                    5 * np.cos(t * 0.3),
                    2 * np.sin(t * 0.7)
                ]
                gyro_mag = np.linalg.norm(self.imu_data["gyroscope"])
                self.imu_motion_magnitude = min(gyro_mag / 200.0, 1.0)
                self.imu_stability = 1.0 / (1.0 + gyro_mag * 0.01)
            
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
                cv2.imshow("Trust Dynamics V3", dashboard)
                
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"trust_v3_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, dashboard)
                print(f"Screenshot saved: {filename}")
                
            # Update tick
            self.tick_count += 1
            
            # Status
            if self.tick_count % 30 == 0:
                print(f"Tick {self.tick_count:4d} | "
                      f"RF: {self.reality_field:.2f} | "
                      f"Context: {self.context_state.name:8s} | "
                      f"Trust: L={self.trust_weights['camera_left']:.2f} "
                      f"R={self.trust_weights['camera_right']:.2f} "
                      f"I={self.trust_weights['imu']:.2f} | "
                      f"Quality: L={self.camera_left_quality:.2f} "
                      f"R={self.camera_right_quality:.2f}")
                      
            # Check for experiment end
            if phase_idx >= len(phases) - 1 and time.time() - phase_start > phases[-1][1]:
                print("\n>>> EXPERIMENT COMPLETE <<<")
                break
                      
        self.shutdown()
        
    def initialize(self):
        """Initialize cameras and IMU"""
        print("\n" + "="*60)
        print("COHERENCE V3 - TRUST DYNAMICS EXPERIMENT")
        print("="*60)
        print("\nInitializing...")
        
        # Initialize cameras
        self.cap_l = cv2.VideoCapture(self.gst_pipeline(0), cv2.CAP_GSTREAMER)
        self.cap_r = cv2.VideoCapture(self.gst_pipeline(1), cv2.CAP_GSTREAMER)
        
        # Test cameras
        ret_l, test_l = self.cap_l.read()
        ret_r, test_r = self.cap_r.read()
        
        if ret_l and ret_r:
            print(f"✓ Both cameras initialized")
        else:
            print(f"✗ Camera issue - Left: {ret_l}, Right: {ret_r}")
        
        # Initialize IMU
        try:
            self.imu = YahboomCMP10A(port="/dev/ttyUSB0", baud=9600)
            self.use_real_imu = True
            
            # Start IMU reading thread
            self.imu_thread = threading.Thread(target=self.read_imu_loop)
            self.imu_thread.daemon = True
            self.imu_thread.start()
            
            print("✓ Real IMU connected")
        except Exception as e:
            print(f"✗ Using simulated IMU: {e}")
            self.use_real_imu = False
            
        print("\n✓ System ready")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("="*60 + "\n")
        
    def create_dashboard(self, frame_l, frame_r):
        """Create visualization dashboard with IMU data"""
        dashboard = np.zeros((720, 1280, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)
        
        # Camera feeds (smaller)
        if frame_l is not None:
            small_l = cv2.resize(frame_l, (320, 180))
            dashboard[20:200, 20:340] = small_l
            # Quality bar
            quality_h = int(160 * self.camera_left_quality)
            cv2.rectangle(dashboard, (350, 200-quality_h), (370, 200), 
                         (0, 255, 0) if self.camera_left_quality > 0.5 else (0, 0, 255), -1)
            
        if frame_r is not None:
            small_r = cv2.resize(frame_r, (320, 180))
            dashboard[20:200, 400:720] = small_r
            # Quality bar
            quality_h = int(160 * self.camera_right_quality)
            cv2.rectangle(dashboard, (730, 200-quality_h), (750, 200),
                         (0, 255, 0) if self.camera_right_quality > 0.5 else (0, 0, 255), -1)
                   
        # Labels
        cv2.putText(dashboard, f"L: {self.camera_left_quality:.2f}",
                   (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(dashboard, f"R: {self.camera_right_quality:.2f}",
                   (410, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                   
        # Reality field visualization
        center_x, center_y = 640, 380
        radius = int(50 * (1 + self.reality_field))
        color_intensity = int(255 * self.reality_field)
        color = (0, color_intensity, 255 - color_intensity)
        cv2.circle(dashboard, (center_x, center_y), radius, color, -1)
        cv2.circle(dashboard, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Status text
        cv2.putText(dashboard, f"Reality Field: {self.reality_field:.3f}",
                   (540, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                   
        # Context state with color
        context_colors = {
            ContextState.STABLE: (0, 255, 0),
            ContextState.MOVING: (255, 255, 0),
            ContextState.UNSTABLE: (255, 165, 0),
            ContextState.NOVEL: (255, 0, 255)
        }
        color = context_colors.get(self.context_state, (255, 255, 255))
        cv2.putText(dashboard, f"Context: {self.context_state.name}",
                   (540, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                   
        # Trust weights
        y = 280
        cv2.putText(dashboard, "Trust Weights:", (900, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 30
        for sensor, trust in self.trust_weights.items():
            color = (0, int(255 * trust), int(255 * (1-trust)))
            cv2.putText(dashboard, f"{sensor:12s}: {trust:.3f}",
                       (900, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # Trust bar
            cv2.rectangle(dashboard, (1100, y-15), (1100 + int(100*trust), y-5), color, -1)
            y += 25
            
        # IMU Data
        y = 420
        imu_label = "IMU (REAL)" if self.use_real_imu else "IMU (SIM)"
        cv2.putText(dashboard, imu_label, (900, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y += 25
        
        # Gyroscope
        gyro = self.imu_data["gyroscope"]
        cv2.putText(dashboard, f"Gyro: [{gyro[0]:6.1f}, {gyro[1]:6.1f}, {gyro[2]:6.1f}]°/s",
                   (900, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 20
        
        # Acceleration
        acc = self.imu_data["acceleration"]
        cv2.putText(dashboard, f"Acc:  [{acc[0]:6.2f}, {acc[1]:6.2f}, {acc[2]:6.2f}]g",
                   (900, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 20
        
        # IMU metrics
        cv2.putText(dashboard, f"IMU Motion: {self.imu_motion_magnitude:.3f}",
                   (900, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
        y += 20
        cv2.putText(dashboard, f"IMU Stability: {self.imu_stability:.3f}",
                   (900, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
        
        # Sensor metrics (left side)
        y = 280
        cv2.putText(dashboard, "Sensor Metrics:", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 30
        cv2.putText(dashboard, f"Camera Motion: {self.camera_motion:.3f}",
                   (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        
        # Phase indicator
        cv2.putText(dashboard, f"Tick: {self.tick_count}",
                   (50, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                   
        return dashboard
        
    def shutdown(self):
        """Clean shutdown"""
        print("\n" + "="*60)
        print("SHUTTING DOWN")
        print("="*60)
        
        self.running = False
        
        # Stop IMU thread
        if self.imu_thread:
            self.imu_thread.join(timeout=1.0)
            
        # Close IMU
        if self.imu and self.imu.ser and self.imu.ser.is_open:
            self.imu.ser.close()
        
        # Finalize logging
        stats = self.logger.finalize()
        analysis = self.logger.analyze()
        
        print(f"\nExperiment Statistics:")
        print(f"  Total entries: {stats['entries_logged']}")
        print(f"  Duration: {analysis.get('duration', 0):.1f}s")
        
        if 'trust_changes' in analysis:
            print(f"\nTrust Changes:")
            for sensor, changes in analysis['trust_changes'].items():
                if 'change' in changes:
                    print(f"  {sensor:12s}: {changes['initial']:.3f} → {changes['final']:.3f} "
                          f"(Δ={changes['change']:+.3f})")
        
        if 'phases' in analysis:
            print(f"\nPhase Analysis:")
            for phase, data in analysis['phases'].items():
                print(f"  {phase:15s}: RF avg={data['avg_reality_field']:.3f}")
        
        # Close cameras
        if self.cap_l:
            self.cap_l.release()
        if self.cap_r:
            self.cap_r.release()
            
        cv2.destroyAllWindows()
        print("\n✓ Shutdown complete")
        

if __name__ == "__main__":
    experiment = CoherenceV3()
    
    def signal_handler(signum, frame):
        print("\n[INTERRUPT] Stopping experiment...")
        experiment.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        experiment.run()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        experiment.shutdown()