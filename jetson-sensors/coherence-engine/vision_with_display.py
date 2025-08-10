#!/usr/bin/env python3
"""
Vision sensor with live display for coherence engine.
Based on our working vision experiments with simplified initialization.
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from collections import deque
import sys

# Add sensors to path
sys.path.insert(0, str(Path(__file__).parent / "sensors"))

class VisionCoherenceDisplay:
    def __init__(self):
        self.running = True
        self.frame_count = 0
        
        # Memory integration
        self.memory_dir = Path("memory/experiences")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Data tracking
        self.field_history = deque(maxlen=100)
        self.sensor_data = deque(maxlen=100)
        
        # Initialize cameras with simple approach
        self.init_cameras()
        
        # Create display window
        cv2.namedWindow('Coherence Vision', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Coherence Vision', 1280, 960)
        
    def init_cameras(self):
        """Initialize cameras with GStreamer pipeline."""
        print("Initializing cameras...")
        
        # GStreamer pipeline for CSI cameras on Jetson
        def gst_pipeline(sensor_id=0):
            return (
                f"nvarguscamerasrc sensor-id={sensor_id} ! "
                f"video/x-raw(memory:NVMM), width=3280, height=2464, format=NV12, framerate=21/1 ! "
                f"nvvidconv ! video/x-raw, width=640, height=480, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink"
            )
        
        # Initialize CSI cameras with GStreamer
        self.cap_l = cv2.VideoCapture(gst_pipeline(0), cv2.CAP_GSTREAMER)
        self.cap_r = cv2.VideoCapture(gst_pipeline(1), cv2.CAP_GSTREAMER)
            
        # Test cameras
        ret_l, _ = self.cap_l.read()
        ret_r, _ = self.cap_r.read()
        
        if ret_l and ret_r:
            print("Dual cameras initialized successfully")
            self.dual_camera = True
        elif ret_l:
            print("Only left camera available")
            self.dual_camera = False
        else:
            print("No cameras available, using test pattern")
            self.dual_camera = False
            
        # For motion detection
        self.prev_gray_l = None
        self.prev_gray_r = None
        
    def generate_test_pattern(self):
        """Generate test pattern when no cameras available."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving gradient
        t = int(time.time() * 10) % 100
        for i in range(0, 640, 20):
            color = int(128 + 127 * np.sin((i + t) * 0.1))
            cv2.line(img, (i, 0), (i, 480), (0, color, color//2), 2)
            
        # Add timestamp
        cv2.putText(img, f"Test Pattern {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img
        
    def compute_vision_confidence(self, frame_l, frame_r=None):
        """Compute vision confidence score for coherence engine."""
        if frame_l is None:
            return 0.0
            
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        confidence = 0.5  # Base confidence
        
        # Motion detection
        if self.prev_gray_l is not None:
            # Optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray_l, gray_l, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion = np.mean(mag)
            
            # Add motion to confidence
            confidence += min(0.3, motion * 0.1)
            
        # Stereo correlation if available
        if frame_r is not None and self.dual_camera:
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            
            # Simple correlation
            if gray_l.shape == gray_r.shape:
                corr = np.corrcoef(gray_l.flatten(), gray_r.flatten())[0, 1]
                if 0.3 < corr < 0.9:  # Good stereo correlation
                    confidence += 0.2
                    
            self.prev_gray_r = gray_r
            
        self.prev_gray_l = gray_l
        return min(1.0, confidence)
        
    def load_coherence_data(self):
        """Load latest coherence data from memory."""
        try:
            date_str = time.strftime("%Y%m%d")
            exp_file = self.memory_dir / f"experiences_{date_str}.json"
            
            if exp_file.exists():
                with open(exp_file, 'r') as f:
                    data = json.load(f)
                    if data:
                        latest = data[-1]
                        self.field_history.append(latest.get('field_value', 0))
                        self.sensor_data.append(latest.get('sensor_readings', {}))
                        return latest
        except:
            pass
        return None
        
    def draw_coherence_overlay(self, canvas, data):
        """Draw coherence information overlay on video."""
        if not data:
            return
            
        # Reality field value
        field_val = data.get('field_value', 0)
        cv2.putText(canvas, f"Reality Field: {field_val:.3f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Context state
        context = data.get('context_state', 'UNKNOWN')
        color = {'STABLE': (0, 255, 0), 'MOVING': (255, 255, 0),
                'UNSTABLE': (0, 165, 255), 'NOVEL': (0, 0, 255)}.get(context, (255, 255, 255))
        cv2.putText(canvas, f"Context: {context}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Sensor values
        sensors = data.get('sensor_readings', {})
        y = 90
        for name, value in sensors.items():
            cv2.putText(canvas, f"{name}: {value:.2f}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            
        # Draw mini graph of reality field
        if len(self.field_history) > 1:
            graph_x, graph_y = 10, 250
            graph_w, graph_h = 200, 100
            
            # Draw border
            cv2.rectangle(canvas, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h),
                         (100, 100, 100), 1)
            
            # Draw graph
            points = []
            for i, val in enumerate(list(self.field_history)[-50:]):
                x = graph_x + int(i * graph_w / 50)
                y = graph_y + graph_h - int(val * graph_h)
                points.append((x, y))
                
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i+1], (0, 255, 255), 2)
                
    def run(self):
        """Main display loop."""
        print("Coherence Vision Display Started")
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        
        while self.running:
            # Get frames
            if self.dual_camera or self.cap_l.isOpened():
                ret_l, frame_l = self.cap_l.read()
                if self.dual_camera:
                    ret_r, frame_r = self.cap_r.read()
                else:
                    ret_r, frame_r = False, None
                    
                if not ret_l:
                    frame_l = self.generate_test_pattern()
                    frame_r = self.generate_test_pattern()
            else:
                frame_l = self.generate_test_pattern()
                frame_r = self.generate_test_pattern()
                
            # Compute vision confidence
            confidence = self.compute_vision_confidence(frame_l, frame_r)
            
            # Create display canvas
            h, w = frame_l.shape[:2]
            
            if self.dual_camera and frame_r is not None:
                # Side by side view with extra space for depth
                canvas = np.zeros((h + 120, w*2 + 20, 3), dtype=np.uint8)
                canvas[:h, :w] = frame_l
                canvas[:h, w+20:] = frame_r
                
                # Center divider
                cv2.line(canvas, (w+10, 0), (w+10, h), (255, 255, 255), 2)
                
                # Stereo difference in bottom strip (no overlay on main cameras)
                diff = cv2.absdiff(frame_l, frame_r)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                
                # Create small depth visualization at bottom
                depth_viz = cv2.resize(diff_gray, (w*2 + 20, 100))
                depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
                canvas[h+10:h+110, :] = depth_colored
                
                # Label the depth strip
                cv2.putText(canvas, "STEREO DEPTH", (10, h+105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            else:
                canvas = frame_l.copy()
                
            # Load and display coherence data
            if self.frame_count % 5 == 0:  # Update every 5 frames
                coherence_data = self.load_coherence_data()
                if coherence_data:
                    coherence_data['vision_confidence'] = confidence
            else:
                coherence_data = None
                
            # Draw overlay
            if self.frame_count > 0:
                self.draw_coherence_overlay(canvas, coherence_data or 
                    {'field_value': 0, 'context_state': 'INITIALIZING',
                     'sensor_readings': {'vision': confidence}})
                
            # Frame counter
            cv2.putText(canvas, f"Frame: {self.frame_count}", 
                       (canvas.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('Coherence Vision', canvas)
            
            # Handle keys
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"coherence_vision_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, canvas)
                print(f"Saved screenshot: {filename}")
                
            self.frame_count += 1
            
        # Cleanup
        self.cap_l.release()
        if self.dual_camera:
            self.cap_r.release()
        cv2.destroyAllWindows()
        print("Vision display stopped")

if __name__ == "__main__":
    display = VisionCoherenceDisplay()
    display.run()