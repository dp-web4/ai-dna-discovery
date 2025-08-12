#!/usr/bin/env python3
"""
Legion Dashboard - Visual coherence engine with camera, audio, and sensor display
"""

import sys
import os
import time
import threading
import queue
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

# Import core engine
from core.engine import CoherenceEngine

# Import Legion sensors
from plugins.legion.gpu_sensor import GPUSensor
from plugins.legion.audio_sensor import AudioSensor
from plugins.legion.camera_sensor import CameraSensor

class LegionDashboard:
    def __init__(self):
        self.camera_sensor = None
        self.audio_sensor = None
        self.gpu_sensor = None
        self.running = True
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Dashboard dimensions
        self.width = 1280
        self.height = 720
        self.sidebar_width = 400
        
        # Initialize sensors first
        self.init_sensors()
        
        # Then create engine with sensors
        sensors = []
        if self.camera_sensor and self.camera_sensor.available:
            sensors.append(self.camera_sensor)
        if self.audio_sensor and self.audio_sensor.available:
            sensors.append(self.audio_sensor)
        if self.gpu_sensor and self.gpu_sensor.available:
            sensors.append(self.gpu_sensor)
        
        # Initialize engine with sensors
        from core.engine import Context
        self.engine = CoherenceEngine(sensors=sensors, context=Context())
        self.tick = 0
        
    def init_sensors(self):
        """Initialize all available sensors"""
        print("\n" + "="*60)
        print("LEGION VISUAL DASHBOARD")
        print("Reality Field Coherence Engine")
        print("="*60)
        print("\nDetecting sensors...")
        
        # Camera sensor
        self.camera_sensor = CameraSensor()
        if self.camera_sensor.available:
            print("  ✓ Camera detected")
            # Test getting a frame
            import time
            time.sleep(1)  # Give it time to capture
            test_frame = self.camera_sensor.get_current_frame()
            if test_frame is not None:
                print(f"    Camera working: {test_frame.shape}")
            else:
                print("    WARNING: Camera detected but no frames")
        else:
            print("  ✗ No camera available")
        
        # Audio sensor
        self.audio_sensor = AudioSensor()
        if self.audio_sensor.available:
            print("  ✓ Audio input detected")
        else:
            print("  ✗ No audio input available")
        
        # GPU sensor
        self.gpu_sensor = GPUSensor()
        if self.gpu_sensor.available:
            print("  ✓ GPU detected (limited without nvidia-smi)")
        else:
            print("  ✗ No GPU detected")
        
        # Check for IMU (not available on Legion)
        print("  ✗ No IMU (not available on Legion)")
        
        print("\nPress 'q' to quit\n")
        
    def create_dashboard(self, frame=None):
        """Create the main dashboard display"""
        # Create base canvas
        dashboard = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Camera feed area (left side)
        camera_width = self.width - self.sidebar_width
        if frame is not None:
            # Resize frame to fit
            try:
                frame_resized = cv2.resize(frame, (camera_width, self.height))
                dashboard[:, :camera_width] = frame_resized
                # Add camera indicator
                cv2.putText(dashboard, "CAMERA LIVE", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Frame resize error: {e}")
                # Show error placeholder
                cv2.putText(dashboard, f"CAMERA ERROR: {str(e)[:30]}", 
                           (camera_width//2 - 200, self.height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            # No camera placeholder
            cv2.putText(dashboard, "NO CAMERA FEED", 
                       (camera_width//2 - 150, self.height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
            cv2.rectangle(dashboard, (10, 10), (camera_width-10, self.height-10),
                         (50, 50, 50), 2)
            if self.camera_sensor and self.camera_sensor.available:
                cv2.putText(dashboard, "(Camera detected but no frames)", 
                           (camera_width//2 - 180, self.height//2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        
        # Sidebar (right side)
        sidebar_x = camera_width
        dashboard[:, sidebar_x:] = (40, 40, 40)  # Dark gray background
        
        # Title
        cv2.putText(dashboard, "COHERENCE ENGINE", 
                   (sidebar_x + 20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Get current engine state
        reality_field = self.engine.context.prev_field_value or 0.0
        context = self.engine.context.state.name
        tick = self.tick
        
        # Reality field visualization
        y_offset = 100
        cv2.putText(dashboard, f"Reality Field: {reality_field:.3f}", 
                   (sidebar_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        
        # Draw reality field bar
        bar_width = int(reality_field * 300)
        cv2.rectangle(dashboard, 
                     (sidebar_x + 20, y_offset + 10),
                     (sidebar_x + 20 + bar_width, y_offset + 30),
                     (0, 255, 255), -1)
        cv2.rectangle(dashboard, 
                     (sidebar_x + 20, y_offset + 10),
                     (sidebar_x + 320, y_offset + 30),
                     (100, 100, 100), 1)
        
        # Context state
        y_offset += 60
        context_color = {
            "STABLE": (0, 255, 0),
            "MOVING": (255, 255, 0),
            "UNSTABLE": (0, 165, 255),
            "NOVEL": (255, 0, 255)
        }.get(context, (255, 255, 255))
        
        cv2.putText(dashboard, f"Context: {context}", 
                   (sidebar_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, context_color, 2)
        
        # Sensor readings
        y_offset += 60
        cv2.putText(dashboard, "SENSORS:", 
                   (sidebar_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 30
        
        # Camera sensor
        if self.camera_sensor and self.camera_sensor.available:
            motion = self.camera_sensor.motion_level
            cv2.putText(dashboard, f"Camera Motion: {motion:.2f}", 
                       (sidebar_x + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Motion bar
            motion_bar = int(motion * 200)
            cv2.rectangle(dashboard, 
                         (sidebar_x + 180, y_offset - 15),
                         (sidebar_x + 180 + motion_bar, y_offset - 5),
                         (0, 255, 0), -1)
            cv2.rectangle(dashboard, 
                         (sidebar_x + 180, y_offset - 15),
                         (sidebar_x + 380, y_offset - 5),
                         (100, 100, 100), 1)
        else:
            cv2.putText(dashboard, "Camera: Not Available", 
                       (sidebar_x + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        y_offset += 35
        
        # Audio sensor with level meter
        if self.audio_sensor and self.audio_sensor.available:
            audio_level = self.audio_sensor.current_level
            cv2.putText(dashboard, f"Audio Level: {audio_level:.2f}", 
                       (sidebar_x + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            
            # Audio level bar with peak indicators
            audio_bar = int(audio_level * 200)
            # Background
            cv2.rectangle(dashboard, 
                         (sidebar_x + 180, y_offset - 15),
                         (sidebar_x + 380, y_offset - 5),
                         (50, 50, 50), -1)
            # Level bar with color gradient
            if audio_level < 0.5:
                bar_color = (0, 255, 0)  # Green
            elif audio_level < 0.8:
                bar_color = (0, 255, 255)  # Yellow
            else:
                bar_color = (0, 0, 255)  # Red
            
            cv2.rectangle(dashboard, 
                         (sidebar_x + 180, y_offset - 15),
                         (sidebar_x + 180 + audio_bar, y_offset - 5),
                         bar_color, -1)
            # Border
            cv2.rectangle(dashboard, 
                         (sidebar_x + 180, y_offset - 15),
                         (sidebar_x + 380, y_offset - 5),
                         (100, 100, 100), 1)
            
            # Add peak markers
            for i in range(0, 11, 2):  # 0, 20%, 40%, 60%, 80%, 100%
                x = sidebar_x + 180 + int(i * 20)
                cv2.line(dashboard, (x, y_offset - 18), (x, y_offset - 2), (150, 150, 150), 1)
        else:
            cv2.putText(dashboard, "Audio: Not Available", 
                       (sidebar_x + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        y_offset += 35
        
        # GPU sensor
        if self.gpu_sensor and self.gpu_sensor.available:
            gpu_util = self.gpu_sensor.last_util if hasattr(self.gpu_sensor, 'last_util') else 0.05
            cv2.putText(dashboard, f"GPU: {gpu_util:.2f}", 
                       (sidebar_x + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            # GPU bar
            gpu_bar = int(gpu_util * 200)
            cv2.rectangle(dashboard, 
                         (sidebar_x + 180, y_offset - 15),
                         (sidebar_x + 180 + gpu_bar, y_offset - 5),
                         (0, 200, 255), -1)
            cv2.rectangle(dashboard, 
                         (sidebar_x + 180, y_offset - 15),
                         (sidebar_x + 380, y_offset - 5),
                         (100, 100, 100), 1)
        else:
            cv2.putText(dashboard, "GPU: Not Available", 
                       (sidebar_x + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        y_offset += 35
        
        # IMU status (always unavailable on Legion)
        cv2.putText(dashboard, "IMU: Not Available (Legion)", 
                   (sidebar_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Tick counter
        y_offset = self.height - 80
        cv2.putText(dashboard, f"Tick: {tick}", 
                   (sidebar_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Timestamp
        y_offset += 25
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(dashboard, timestamp, 
                   (sidebar_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Instructions
        y_offset += 25
        cv2.putText(dashboard, "Press 'q' to quit", 
                   (sidebar_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        return dashboard
    
    def engine_thread(self):
        """Run the coherence engine in a separate thread"""
        while self.running:
            # Step the engine
            field_value = self.engine.step(tick=self.tick)
            self.tick += 1
            
            # Get camera frame if available
            if self.camera_sensor and self.camera_sensor.available:
                frame = self.camera_sensor.get_current_frame()
                if frame is not None:
                    try:
                        # Clear old frames
                        while not self.frame_queue.empty():
                            try:
                                self.frame_queue.get_nowait()
                            except:
                                break
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
            
            time.sleep(0.1)  # 10 Hz update rate
    
    def run(self):
        """Main dashboard loop"""
        # Start engine thread
        engine_thread = threading.Thread(target=self.engine_thread)
        engine_thread.daemon = True
        engine_thread.start()
        
        # Create window
        cv2.namedWindow('Legion Coherence Dashboard', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Legion Coherence Dashboard', self.width, self.height)
        
        current_frame = None
        frame_count = 0
        
        while self.running:
            # Try to get latest frame
            try:
                new_frame = self.frame_queue.get_nowait()
                if new_frame is not None:
                    current_frame = new_frame
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"Dashboard: Received frame {frame_count}, shape: {current_frame.shape}")
            except queue.Empty:
                pass
            
            # Create and display dashboard
            dashboard = self.create_dashboard(current_frame)
            cv2.imshow('Legion Coherence Dashboard', dashboard)
            
            # Check for quit
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                self.running = False
                break
        
        # Cleanup
        cv2.destroyAllWindows()
        print("\nDashboard stopped.")

if __name__ == "__main__":
    dashboard = LegionDashboard()
    dashboard.run()