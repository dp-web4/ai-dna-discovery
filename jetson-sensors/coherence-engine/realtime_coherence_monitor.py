#!/usr/bin/env python3
"""
Real-time coherence monitor with video feed and live graphs.
Shows dual camera feeds alongside reality field visualization.
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from collections import deque
import threading

class CoherenceMonitor:
    def __init__(self):
        self.running = True
        self.camera_width = 640
        self.camera_height = 480
        
        # Data storage for graphs
        self.field_history = deque(maxlen=100)
        self.sensor_history = {
            'vision': deque(maxlen=100),
            'imu': deque(maxlen=100),
            'memory': deque(maxlen=100),
            'cognition': deque(maxlen=100)
        }
        
        # Initialize cameras
        self.init_cameras()
        
        # Create windows
        cv2.namedWindow('Coherence Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Coherence Monitor', 1920, 1080)
        
    def init_cameras(self):
        """Initialize CSI cameras."""
        def gst_pipeline(sensor_id=0):
            return (
                f"nvarguscamerasrc sensor-id={sensor_id} ! "
                f"video/x-raw(memory:NVMM), width=3280, height=2464, format=NV12, framerate=21/1 ! "
                f"nvvidconv ! video/x-raw, width={self.camera_width}, height={self.camera_height}, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink"
            )
        
        self.cap_l = cv2.VideoCapture(gst_pipeline(0), cv2.CAP_GSTREAMER)
        self.cap_r = cv2.VideoCapture(gst_pipeline(1), cv2.CAP_GSTREAMER)
        
    def load_latest_data(self):
        """Load latest coherence data from memory."""
        try:
            # Check for today's experience file
            memory_dir = Path("memory/experiences")
            if memory_dir.exists():
                date_str = time.strftime("%Y%m%d")
                exp_file = memory_dir / f"experiences_{date_str}.json"
                if exp_file.exists():
                    with open(exp_file, 'r') as f:
                        data = json.load(f)
                        if data:
                            latest = data[-1]
                            # Add to history
                            self.field_history.append(latest.get('field_value', 0.5))
                            for sensor in ['vision', 'imu', 'memory', 'cognition']:
                                if sensor in latest.get('sensor_readings', {}):
                                    self.sensor_history[sensor].append(
                                        latest['sensor_readings'][sensor]
                                    )
        except Exception as e:
            pass
            
    def draw_graph(self, img, data, x, y, w, h, title, color=(0, 255, 0)):
        """Draw a graph on the image."""
        # Draw border
        cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 100), 2)
        
        # Title
        cv2.putText(img, title, (x+5, y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if len(data) > 1:
            # Scale data to fit
            data_array = np.array(list(data))
            data_min = 0  # Always use 0-1 range for sensors
            data_max = 1
            
            # Draw grid lines
            for i in range(5):
                grid_y = y + h - int(i * h / 4)
                cv2.line(img, (x, grid_y), (x+w, grid_y), (50, 50, 50), 1)
                
            # Draw data
            points = []
            for i, val in enumerate(data_array):
                px = x + int(i * w / len(data_array))
                py = y + h - int((val - data_min) / (data_max - data_min) * h)
                py = max(y, min(y+h, py))  # Clamp to graph area
                points.append((px, py))
                
            # Draw line
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i+1], color, 2)
                
            # Show current value
            if data:
                current = data[-1]
                cv2.putText(img, f"{current:.3f}", (x+w-60, y+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
    def draw_sensor_bars(self, img, x, y, w, h):
        """Draw sensor contribution bars."""
        # Draw border
        cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 100), 2)
        
        # Title
        cv2.putText(img, "Sensor Contributions", (x+5, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Get latest values
        sensors = ['vision', 'imu', 'memory', 'cognition']
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        
        bar_width = w // 5
        bar_spacing = bar_width // 4
        
        for i, (sensor, color) in enumerate(zip(sensors, colors)):
            if self.sensor_history[sensor]:
                value = self.sensor_history[sensor][-1]
                
                # Draw bar
                bar_x = x + bar_spacing + i * (bar_width + bar_spacing)
                bar_height = int(value * (h - 40))
                bar_y = y + h - bar_height - 20
                
                cv2.rectangle(img, (bar_x, bar_y), 
                            (bar_x + bar_width, y + h - 20),
                            color, -1)
                
                # Label
                cv2.putText(img, sensor[:3].upper(), 
                           (bar_x + 5, y + h - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Value
                cv2.putText(img, f"{value:.2f}",
                           (bar_x + 5, bar_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                           
    def run(self):
        """Main monitoring loop."""
        print("Coherence Monitor Started")
        print("Press 'q' to quit")
        
        frame_count = 0
        
        while self.running:
            # Read cameras
            ret_l, frame_l = self.cap_l.read()
            ret_r, frame_r = self.cap_r.read()
            
            if not ret_l or not ret_r:
                continue
                
            # Create display canvas
            canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            # Place camera feeds (top half)
            # Left camera
            frame_l_resized = cv2.resize(frame_l, (640, 480))
            canvas[50:530, 50:690] = frame_l_resized
            cv2.putText(canvas, "LEFT CAMERA", (50, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Right camera
            frame_r_resized = cv2.resize(frame_r, (640, 480))
            canvas[50:530, 750:1390] = frame_r_resized
            cv2.putText(canvas, "RIGHT CAMERA", (750, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Stereo difference view
            diff = cv2.absdiff(frame_l_resized, frame_r_resized)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            diff_color = cv2.cvtColor(diff_thresh, cv2.COLOR_GRAY2BGR)
            diff_color[:,:,1] = diff_thresh  # Make it green
            
            canvas[50:530, 1440:1870] = cv2.resize(diff_color, (430, 480))
            cv2.putText(canvas, "STEREO DIFF", (1440, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Load latest data every 10 frames
            if frame_count % 10 == 0:
                self.load_latest_data()
            
            # Draw graphs (bottom half)
            # Reality field graph
            self.draw_graph(canvas, self.field_history, 
                          50, 580, 600, 400,
                          "Reality Field", (0, 255, 255))
            
            # Sensor contributions
            self.draw_sensor_bars(canvas, 700, 580, 500, 400)
            
            # Individual sensor graphs
            self.draw_graph(canvas, self.sensor_history['vision'],
                          1250, 580, 300, 180,
                          "Vision", (0, 255, 0))
            self.draw_graph(canvas, self.sensor_history['imu'],
                          1570, 580, 300, 180,
                          "IMU", (255, 0, 0))
            self.draw_graph(canvas, self.sensor_history['memory'],
                          1250, 780, 300, 180,
                          "Memory", (0, 0, 255))
            self.draw_graph(canvas, self.sensor_history['cognition'],
                          1570, 780, 300, 180,
                          "Cognition", (255, 255, 0))
            
            # Status text
            cv2.putText(canvas, f"Frame: {frame_count}", (50, 1050),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if self.field_history:
                cv2.putText(canvas, f"Reality Field: {self.field_history[-1]:.3f}", 
                          (250, 1050),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Display
            cv2.imshow('Coherence Monitor', canvas)
            
            # Check for quit
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
                
            frame_count += 1
            
        # Cleanup
        self.cap_l.release()
        self.cap_r.release()
        cv2.destroyAllWindows()
        print("Monitor stopped")

if __name__ == "__main__":
    monitor = CoherenceMonitor()
    monitor.run()