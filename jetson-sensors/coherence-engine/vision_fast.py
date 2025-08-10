#!/usr/bin/env python3
"""
Fast vision display optimized for high frame rate.
Minimal processing to match binocular test performance.
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from collections import deque
import threading
import queue

class FastVisionDisplay:
    def __init__(self):
        self.running = True
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
        # Coherence data queue for async updates
        self.data_queue = queue.Queue()
        self.latest_coherence = None
        
        # Initialize cameras with minimal buffering
        self.init_cameras()
        
        # Start data loader thread
        self.data_thread = threading.Thread(target=self.data_loader, daemon=True)
        self.data_thread.start()
        
        # Create display window
        cv2.namedWindow('Fast Coherence Vision', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fast Coherence Vision', 1280, 720)
        
    def init_cameras(self):
        """Initialize cameras with minimal latency pipeline."""
        print("Initializing high-speed cameras at 1080p...")
        
        def gst_pipeline(sensor_id=0):
            # Force both cameras to exact same 1080p mode
            # sensor-mode=2 explicitly selects 1920x1080 @ 30fps mode
            return (
                f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
                f"video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
                f"nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! "  # Keep aspect ratio
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink drop=true max-buffers=1 sync=false"  # No sync for lowest latency
            )
        
        self.cap_l = cv2.VideoCapture(gst_pipeline(0), cv2.CAP_GSTREAMER)
        self.cap_r = cv2.VideoCapture(gst_pipeline(1), cv2.CAP_GSTREAMER)
        
        # Set minimal buffering
        for cap in [self.cap_l, self.cap_r]:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        print("Cameras initialized for 30 FPS operation")
        
    def data_loader(self):
        """Background thread to load coherence data without blocking video."""
        memory_dir = Path("memory/experiences")
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        while self.running:
            try:
                date_str = time.strftime("%Y%m%d")
                exp_file = memory_dir / f"experiences_{date_str}.json"
                
                if exp_file.exists():
                    with open(exp_file, 'r') as f:
                        data = json.load(f)
                        if data:
                            self.latest_coherence = data[-1]
                            
            except:
                pass
                
            time.sleep(0.5)  # Update coherence data 2x per second
            
    def draw_minimal_overlay(self, frame, is_left=True):
        """Draw minimal overlay to maintain high FPS."""
        # FPS counter
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 30
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Camera label
        label = "LEFT" if is_left else "RIGHT"
        cv2.putText(frame, label, 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Minimal coherence data (only if available and left camera)
        if is_left and self.latest_coherence:
            field = self.latest_coherence.get('field_value', 0)
            context = self.latest_coherence.get('context_state', 'UNKNOWN')
            
            cv2.putText(frame, f"Field: {field:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Ctx: {context}", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                       
    def run(self):
        """Main high-speed display loop."""
        print("Fast Vision Display Started - Optimized for 30 FPS")
        print("Press 'q' to quit, 'd' to toggle depth view")
        
        show_depth = True
        
        while self.running:
            # Read frames with no processing delay
            ret_l, frame_l = self.cap_l.read()
            ret_r, frame_r = self.cap_r.read()
            
            if not ret_l or not ret_r:
                continue
                
            # Minimal overlay
            self.draw_minimal_overlay(frame_l, True)
            self.draw_minimal_overlay(frame_r, False)
            
            # Create display
            h, w = frame_l.shape[:2]
            
            if show_depth:
                # Include depth visualization
                canvas = np.zeros((h + 100, w*2 + 20, 3), dtype=np.uint8)
                canvas[:h, :w] = frame_l
                canvas[:h, w+20:] = frame_r
                
                # Fast depth calculation (every 5th frame only)
                if self.frame_count % 5 == 0:
                    diff = cv2.absdiff(frame_l, frame_r)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    self.depth_viz = cv2.resize(diff_gray, (w*2 + 20, 90))
                    
                if hasattr(self, 'depth_viz'):
                    depth_colored = cv2.applyColorMap(self.depth_viz, cv2.COLORMAP_JET)
                    canvas[h+5:h+95, :] = depth_colored
                    cv2.putText(canvas, "DEPTH", (10, h+90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Just side by side
                canvas = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
                canvas[:, :w] = frame_l
                canvas[:, w+20:] = frame_r
                
            # Center line
            cv2.line(canvas, (w+10, 0), (w+10, h), (255, 255, 255), 1)
            
            # Display immediately
            cv2.imshow('Fast Coherence Vision', canvas)
            
            # Handle keys without delay
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_depth = not show_depth
                print(f"Depth view: {'ON' if show_depth else 'OFF'}")
                
            # Update timing
            self.last_time = time.time()
            self.frame_count += 1
            
        # Cleanup
        self.running = False
        self.cap_l.release()
        self.cap_r.release()
        cv2.destroyAllWindows()
        print(f"Stopped. Average FPS: {np.mean(self.fps_history):.1f}")

if __name__ == "__main__":
    display = FastVisionDisplay()
    display.run()