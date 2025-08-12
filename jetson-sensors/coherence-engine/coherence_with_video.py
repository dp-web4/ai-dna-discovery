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

# Import the existing IMU module
sys.path.append('/home/sprout/ai-workspace/private-context/ai-dna-discovery/imu')
from yahboom_cmp10a import YahboomCMP10A

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
        
        # IMU setup
        self.imu = None
        self.imu_thread = None
        self.use_real_imu = False
        
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
        self.imu_data = {
            "acceleration": [0.0, 0.0, 9.81],
            "gyroscope": [0.0, 0.0, 0.0],
            "magnetometer": [30.0, -10.0, 45.0],
            "orientation": [0.0, 0.0, 0.0],
            "temperature": 25.0
        }
        
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
            
        # Try to initialize real IMU
        print("\nInitializing IMU...")
        try:
            self.imu = YahboomCMP10A(port="/dev/ttyUSB0", baud=921600)
            self.use_real_imu = True
            print("✓ Real Yahboom CMP10A IMU connected")
            
            # Start IMU reading thread
            self.imu_thread = threading.Thread(target=self.read_imu_loop)
            self.imu_thread.daemon = True
            self.imu_thread.start()
            
        except Exception as e:
            print(f"✗ Real IMU not available: {e}")
            print("  Using simulated IMU data")
            self.use_real_imu = False
            
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
        
    def read_imu_loop(self):
        """Background thread to read real IMU data from Yahboom CMP10A"""
        while self.running and self.imu:
            try:
                # Read available data into buffer
                data = self.imu.ser.read(self.imu.ser.in_waiting or 1)
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
                print(f"IMU read error: {e}")
                
            time.sleep(0.001)  # 1000Hz potential read rate
    
    def get_imu_data(self):
        """Get current IMU data (real or simulated)"""
        if self.use_real_imu:
            # Return real IMU data
            return self.imu_data
        else:
            # Return simulated data
            return self.simulate_imu()
    
    def simulate_imu(self):
        """Simulate IMU data - updates imu_data dict"""
        t = time.time()
        
        # Update IMU data with simulated values
        self.imu_data["acceleration"] = [
            0.05 * np.sin(t * 2),           # X
            0.05 * np.cos(t * 1.5),         # Y  
            9.81 + 0.1 * np.sin(t)          # Z (gravity + variation)
        ]
        
        self.imu_data["gyroscope"] = [
            0.1 * np.sin(t * 0.5),           # Roll rate
            0.05 * np.cos(t * 0.3),          # Pitch rate
            0.02 * np.sin(t * 0.7)           # Yaw rate
        ]
        
        self.imu_data["magnetometer"] = [
            30 + 5 * np.sin(t * 0.1),       # X
            -10 + 3 * np.cos(t * 0.15),     # Y
            45                               # Z
        ]
        
        # Calculate orientation from accel and mag
        acc = self.imu_data["acceleration"]
        mag = self.imu_data["magnetometer"]
        self.imu_data["orientation"] = [
            np.degrees(np.arctan2(acc[1], acc[2])),     # Roll
            np.degrees(np.arctan2(acc[0], np.sqrt(acc[1]**2 + acc[2]**2))),     # Pitch
            np.degrees(np.arctan2(mag[1], mag[0]))      # Yaw
        ]
        
        self.imu_data["temperature"] = 25 + 5 * np.sin(t * 0.01)
        
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
                   
        # Camera metrics (left side)
        y_pos = 580
        cv2.putText(dashboard, "Camera:", (50, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 25
        
        cv2.putText(dashboard, f"Motion: {self.camera_motion:.3f}",
                   (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        # IMU Full Data (left side, below camera)
        y_pos += 10
        imu_label = "IMU Data (REAL):" if self.use_real_imu else "IMU Data (SIM):"
        imu_color = (0, 255, 0) if self.use_real_imu else (255, 255, 0)
        cv2.putText(dashboard, imu_label, (50, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, imu_color, 2)
        y_pos += 25
        
        if hasattr(self, 'imu_data'):
            # Acceleration
            acc = self.imu_data["acceleration"]
            cv2.putText(dashboard, f"Accel: X:{acc[0]:+.2f} Y:{acc[1]:+.2f} Z:{acc[2]:+.2f}",
                       (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 150), 1)
            y_pos += 20
            
            # Gyroscope
            gyro = self.imu_data["gyroscope"]
            cv2.putText(dashboard, f"Gyro:  X:{gyro[0]:+.2f} Y:{gyro[1]:+.2f} Z:{gyro[2]:+.2f}",
                       (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 200), 1)
            y_pos += 20
            
            # Magnetometer
            mag = self.imu_data["magnetometer"]
            cv2.putText(dashboard, f"Mag:   X:{mag[0]:+.1f} Y:{mag[1]:+.1f} Z:{mag[2]:+.1f}",
                       (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 150), 1)
            y_pos += 20
            
            # Orientation
            orient = self.imu_data["orientation"]
            cv2.putText(dashboard, f"Orient: R:{orient[0]:+.1f}° P:{orient[1]:+.1f}° Y:{orient[2]:+.1f}°",
                       (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
            y_pos += 20
            
            # Temperature
            cv2.putText(dashboard, f"Temp: {self.imu_data['temperature']:.1f}°C",
                       (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_pos += 20
            
            # Computed stability
            cv2.putText(dashboard, f"Stability: {self.imu_stability:.3f}",
                       (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Coherence Interpretation (right side)
        y_pos = 580
        cv2.putText(dashboard, "CE Interpretation:", (1500, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 25
        
        # Show weighted contributions
        camera_weighted = self.camera_motion * self.trust_weights['camera'] * self.relevance_weights['camera']
        imu_weighted = self.imu_stability * self.trust_weights['imu'] * self.relevance_weights['imu']
        
        cv2.putText(dashboard, f"Camera Contrib: {camera_weighted:.3f}",
                   (1500, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        cv2.putText(dashboard, f"IMU Contrib: {imu_weighted:.3f}",
                   (1500, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 25
        
        # Trust weights
        cv2.putText(dashboard, "Trust Weights:", (1500, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 20
        
        cv2.putText(dashboard, f"Camera: {self.trust_weights['camera']:.3f}",
                   (1500, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        cv2.putText(dashboard, f"IMU: {self.trust_weights['imu']:.3f}",
                   (1500, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 25
        
        # Relevance weights
        cv2.putText(dashboard, "Relevance Weights:", (1500, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 20
        
        cv2.putText(dashboard, f"Camera: {self.relevance_weights['camera']:.3f}",
                   (1500, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        cv2.putText(dashboard, f"IMU: {self.relevance_weights['imu']:.3f}",
                   (1500, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
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
            
            # Update IMU data (real or simulated)
            if not self.use_real_imu:
                # Simulate IMU data - this updates self.imu_data
                self.simulate_imu()
            # else: real IMU updates via background thread
            
            # Calculate stability from current IMU data
            gyro_mag = np.linalg.norm(self.imu_data["gyroscope"])
            self.imu_stability = 1.0 / (1.0 + gyro_mag * 10)
            
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
        
        # Stop IMU thread
        if self.imu_thread:
            self.imu_thread.join(timeout=1.0)
            
        # Close IMU connection
        if self.imu and self.imu.ser and self.imu.ser.is_open:
            self.imu.ser.close()
            
        # Release cameras
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