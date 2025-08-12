#!/usr/bin/env python3
"""
Coherence Engine with Working Video Display
Combines the working camera display with coherence engine
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

# Context states
class ContextState(Enum):
    STABLE = auto()
    MOVING = auto()
    UNSTABLE = auto()
    NOVEL = auto()

class CoherenceWithVideo:
    def __init__(self):
        # Camera setup
        self.cap_l = None
        self.cap_r = None
        
        # Coherence state
        self.reality_field = 0.5
        self.context_state = ContextState.STABLE
        self.trust_weights = {"camera": 1.0, "imu": 1.0}
        self.relevance_weights = {"camera": 1.0, "imu": 1.0}
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.tick_count = 0
        self.start_time = time.time()
        self.last_time = time.time()
        
        # Sensor data
        self.camera_motion = 0.0
        self.imu_stability = 0.9
        self.field_history = deque(maxlen=100)
        
        # System control
        self.running = True
        
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
        
    def initialize(self):
        """Initialize cameras and window"""
        print("\n" + "="*60)
        print("COHERENCE ENGINE WITH VIDEO")
        print("="*60)
        
        # Initialize cameras
        print("\nInitializing cameras...")
        self.cap_l = cv2.VideoCapture(self.gst_pipeline(0), cv2.CAP_GSTREAMER)
        self.cap_r = cv2.VideoCapture(self.gst_pipeline(1), cv2.CAP_GSTREAMER)
        
        # Verify cameras
        ret_l, test_l = self.cap_l.read()
        ret_r, test_r = self.cap_r.read()
        
        if ret_l and ret_r:
            print(f"✓ Both cameras initialized")
            print(f"  Left: {test_l.shape}, Right: {test_r.shape}")
        else:
            print(f"✗ Camera issue - Left: {ret_l}, Right: {ret_r}")
            
        # Create window
        self.window_name = "Coherence Engine Dashboard"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)
        
        print("\n✓ System initialized!")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'r' - Reset trust weights")
        print("  Ctrl+C - Emergency stop")
        print("\n" + "="*60 + "\n")
        
    def compute_camera_motion(self, frame_l, frame_r):
        """Compute motion from camera frames"""
        if frame_l is None or frame_r is None:
            return 0.0
            
        # Convert to grayscale
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        
        # Simple motion detection via Laplacian variance
        lap_l = cv2.Laplacian(gray_l, cv2.CV_64F).var()
        lap_r = cv2.Laplacian(gray_r, cv2.CV_64F).var()
        
        # Normalize to 0-1 range
        motion = min((lap_l + lap_r) / 2000.0, 1.0)
        return motion
        
    def simulate_imu(self):
        """Simulate IMU data"""
        t = time.time()
        
        # Simulate stability (inverse of motion)
        base_stability = 0.8 + 0.2 * np.sin(t * 0.5)
        
        # Add some noise
        noise = np.random.normal(0, 0.05)
        
        return max(0, min(1, base_stability + noise))
        
    def update_context(self):
        """Update context based on sensor data"""
        # High motion = MOVING
        if self.camera_motion > 0.3:
            self.context_state = ContextState.MOVING
        # Low stability = UNSTABLE
        elif self.imu_stability < 0.5:
            self.context_state = ContextState.UNSTABLE
        # High variance in history = NOVEL
        elif len(self.field_history) > 10:
            recent_std = np.std(list(self.field_history)[-10:])
            if recent_std > 0.3:
                self.context_state = ContextState.NOVEL
            else:
                self.context_state = ContextState.STABLE
        else:
            self.context_state = ContextState.STABLE
            
    def update_weights(self):
        """Update relevance weights based on context"""
        if self.context_state == ContextState.STABLE:
            self.relevance_weights = {"camera": 1.0, "imu": 0.5}
        elif self.context_state == ContextState.MOVING:
            self.relevance_weights = {"camera": 0.8, "imu": 0.8}
        elif self.context_state == ContextState.UNSTABLE:
            self.relevance_weights = {"camera": 0.5, "imu": 1.0}
        else:  # NOVEL
            self.relevance_weights = {"camera": 1.0, "imu": 1.0}
            
    def update_trust(self):
        """Update trust based on sensor agreement"""
        # Check if sensors agree
        camera_active = self.camera_motion > 0.2
        imu_stable = self.imu_stability > 0.7
        
        if camera_active and not imu_stable:
            # Disagreement - reduce trust
            self.trust_weights["camera"] *= 0.99
            self.trust_weights["imu"] *= 0.99
        elif not camera_active and imu_stable:
            # Agreement - increase trust
            self.trust_weights["camera"] = min(1.0, self.trust_weights["camera"] * 1.01)
            self.trust_weights["imu"] = min(1.0, self.trust_weights["imu"] * 1.01)
            
    def compute_reality_field(self):
        """Compute reality field from weighted sensors"""
        # Get weighted contributions
        camera_contrib = (self.camera_motion * 
                         self.trust_weights["camera"] * 
                         self.relevance_weights["camera"])
        imu_contrib = (self.imu_stability * 
                      self.trust_weights["imu"] * 
                      self.relevance_weights["imu"])
                      
        # Normalize
        total_weight = (self.trust_weights["camera"] * self.relevance_weights["camera"] +
                       self.trust_weights["imu"] * self.relevance_weights["imu"])
                       
        if total_weight > 0:
            self.reality_field = (camera_contrib + imu_contrib) / total_weight
        else:
            self.reality_field = 0.5
            
        # Clamp to valid range
        self.reality_field = max(0.0, min(1.0, self.reality_field))
        
        # Add to history
        self.field_history.append(self.reality_field)
        
    def create_dashboard(self, frame_l, frame_r):
        """Create the dashboard display"""
        # Create base canvas
        dashboard = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)  # Dark gray background
        
        # Draw camera feeds at top
        if frame_l is not None:
            dashboard[20:560, 20:980] = frame_l
            cv2.putText(dashboard, "Left Camera", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                       
        if frame_r is not None:
            dashboard[20:560, 940:1900] = frame_r
            cv2.putText(dashboard, "Right Camera", (950, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                       
        # Draw reality field (center bottom)
        center_x, center_y = 960, 750
        radius = int(100 * (1 + self.reality_field))
        color_intensity = int(255 * self.reality_field)
        color = (0, color_intensity, 255 - color_intensity)
        
        cv2.circle(dashboard, (center_x, center_y), radius, color, -1)
        cv2.circle(dashboard, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Add text overlays
        cv2.putText(dashboard, f"Reality Field: {self.reality_field:.3f}",
                   (center_x - 120, center_y - 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                   
        # Context state with color
        context_colors = {
            ContextState.STABLE: (0, 255, 0),
            ContextState.MOVING: (255, 255, 0),
            ContextState.UNSTABLE: (255, 165, 0),
            ContextState.NOVEL: (255, 0, 255)
        }
        color = context_colors.get(self.context_state, (255, 255, 255))
        cv2.putText(dashboard, f"Context: {self.context_state.name}",
                   (center_x - 100, center_y + 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                   
        # Sensor metrics (left side)
        y_pos = 600
        cv2.putText(dashboard, "Sensor Metrics:", (50, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 40
        
        cv2.putText(dashboard, f"Camera Motion: {self.camera_motion:.3f}",
                   (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 30
        
        cv2.putText(dashboard, f"IMU Stability: {self.imu_stability:.3f}",
                   (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 30
        
        # Trust weights (right side)
        y_pos = 600
        cv2.putText(dashboard, "Trust Weights:", (1600, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 40
        
        cv2.putText(dashboard, f"Camera: {self.trust_weights['camera']:.3f}",
                   (1600, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 30
        
        cv2.putText(dashboard, f"IMU: {self.trust_weights['imu']:.3f}",
                   (1600, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # FPS counter
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            cv2.putText(dashboard, f"FPS: {avg_fps:.1f}",
                       (1800, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                       
        # Tick counter
        cv2.putText(dashboard, f"Tick: {self.tick_count}",
                   (50, 1050), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                   
        return dashboard
        
    def run(self):
        """Main loop"""
        print("Starting coherence engine with video...")
        
        while self.running:
            # Read camera frames
            ret_l, frame_l = self.cap_l.read()
            ret_r, frame_r = self.cap_r.read()
            
            if not ret_l or not ret_r:
                print(f"Frame read error - L:{ret_l} R:{ret_r}")
                continue
                
            # Compute sensor data
            self.camera_motion = self.compute_camera_motion(frame_l, frame_r)
            self.imu_stability = self.simulate_imu()
            
            # Update coherence engine
            self.update_context()
            self.update_weights()
            self.update_trust()
            self.compute_reality_field()
            
            # Create and display dashboard
            dashboard = self.create_dashboard(frame_l, frame_r)
            cv2.imshow(self.window_name, dashboard)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"coherence_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, dashboard)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                self.trust_weights = {"camera": 1.0, "imu": 1.0}
                print("Trust weights reset")
                
            # Update FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time + 0.001)
            self.fps_history.append(fps)
            self.last_time = current_time
            
            # Update tick
            self.tick_count += 1
            
            # Print status every second
            if self.tick_count % 30 == 0:
                elapsed = time.time() - self.start_time
                avg_rate = self.tick_count / elapsed
                print(f"Tick {self.tick_count} | "
                      f"Rate: {avg_rate:.1f} Hz | "
                      f"Reality: {self.reality_field:.3f} | "
                      f"Context: {self.context_state.name}")
                      
        self.shutdown()
        
    def shutdown(self):
        """Clean shutdown"""
        print("\n\nShutting down...")
        self.running = False
        
        if self.cap_l:
            self.cap_l.release()
        if self.cap_r:
            self.cap_r.release()
            
        cv2.destroyAllWindows()
        
        # Print statistics
        total_time = time.time() - self.start_time
        print(f"\nSession Statistics:")
        print(f"  Total ticks: {self.tick_count}")
        print(f"  Runtime: {total_time:.1f} seconds")
        print(f"  Average rate: {self.tick_count/total_time:.1f} Hz")
        print(f"  Final reality field: {self.reality_field:.3f}")
        print(f"  Final context: {self.context_state.name}")
        print("\n✓ Shutdown complete")
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n[INTERRUPT] Received shutdown signal")
        self.running = False


def main():
    # Create and run system
    system = CoherenceWithVideo()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, system.signal_handler)
    
    try:
        system.initialize()
        system.run()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.shutdown()


if __name__ == "__main__":
    main()