#!/usr/bin/env python3
"""
Full 1080p vision display with both cameras at exact same settings.
This should fix blur issues from lane mismatch.
"""

import cv2
import numpy as np
import time
from collections import deque

class Vision1080p:
    def __init__(self):
        self.running = True
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
        # Initialize both cameras at exact same 1080p mode
        self.init_cameras()
        
        # Create display window
        cv2.namedWindow('1080p Stereo Vision', cv2.WINDOW_NORMAL)
        # Set initial window size to fit both cameras
        cv2.resizeWindow('1080p Stereo Vision', 1920, 600)
        
    def init_cameras(self):
        """Initialize both cameras with identical 1080p settings."""
        print("Initializing cameras at 1080p (1920x1080 @ 30fps)...")
        print("Using sensor-mode=2 for both cameras to ensure identical configuration")
        
        def gst_pipeline(sensor_id=0):
            # Explicit sensor-mode=2 for 1920x1080 @ 30fps
            # Both cameras use exact same pipeline
            return (
                f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
                f"video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
                f"nvvidconv ! video/x-raw, width=1920, height=1080, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink drop=true max-buffers=1 sync=false"
            )
        
        # Initialize both cameras with identical settings
        print("Initializing LEFT camera (sensor-id=0)...")
        self.cap_l = cv2.VideoCapture(gst_pipeline(0), cv2.CAP_GSTREAMER)
        
        print("Initializing RIGHT camera (sensor-id=1)...")
        self.cap_r = cv2.VideoCapture(gst_pipeline(1), cv2.CAP_GSTREAMER)
        
        # Verify both initialized
        ret_l, test_l = self.cap_l.read()
        ret_r, test_r = self.cap_r.read()
        
        if ret_l and ret_r:
            print(f"✓ Both cameras initialized successfully")
            print(f"  Left camera shape: {test_l.shape}")
            print(f"  Right camera shape: {test_r.shape}")
        else:
            print(f"✗ Camera initialization issue - Left: {ret_l}, Right: {ret_r}")
            
    def run(self):
        """Main display loop at 1080p."""
        print("\n1080p Stereo Vision Started")
        print("Controls:")
        print("  'q' - Quit")
        print("  'd' - Toggle depth/disparity view")
        print("  's' - Save screenshot")
        print("  'b' - Toggle blur detection")
        
        show_depth = False
        show_blur = False
        
        while self.running:
            # Read frames
            ret_l, frame_l = self.cap_l.read()
            ret_r, frame_r = self.cap_r.read()
            
            if not ret_l or not ret_r:
                print(f"Frame read error - L:{ret_l} R:{ret_r}")
                continue
                
            # Resize frames to fit on screen better (half size)
            frame_l = cv2.resize(frame_l, (960, 540))
            frame_r = cv2.resize(frame_r, (960, 540))
            
            # Calculate FPS
            current_time = time.time()
            if self.last_time:
                fps = 1.0 / (current_time - self.last_time)
                self.fps_history.append(fps)
            self.last_time = current_time
            
            # Blur detection (Laplacian variance)
            if show_blur:
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                blur_l = cv2.Laplacian(gray_l, cv2.CV_64F).var()
                blur_r = cv2.Laplacian(gray_r, cv2.CV_64F).var()
                
                # Overlay blur scores
                cv2.putText(frame_l, f"Sharpness: {blur_l:.1f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_r, f"Sharpness: {blur_r:.1f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if blur_r > 100 else (0, 0, 255), 2)
            
            # FPS overlay
            if self.fps_history:
                avg_fps = np.mean(self.fps_history)
                cv2.putText(frame_l, f"FPS: {avg_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Camera labels
            cv2.putText(frame_l, "LEFT", (10, frame_l.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_r, "RIGHT", (10, frame_r.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Create display
            h, w = frame_l.shape[:2]
            
            if show_depth:
                # With depth visualization
                canvas = np.zeros((h + 150, w*2 + 20, 3), dtype=np.uint8)
                canvas[:h, :w] = frame_l
                canvas[:h, w+20:] = frame_r
                
                # Stereo disparity
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                
                # Simple block matching
                stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
                disparity = stereo.compute(gray_l, gray_r)
                
                # Normalize and colorize
                disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                disparity_color = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
                
                # Resize and add to canvas
                disparity_small = cv2.resize(disparity_color, (w*2 + 20, 140))
                canvas[h+5:h+145, :] = disparity_small
                
                cv2.putText(canvas, "STEREO DEPTH (Blue=Far, Red=Near)", 
                           (10, h+140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                # Simple side by side
                canvas = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
                canvas[:, :w] = frame_l
                canvas[:, w+20:] = frame_r
                
                # Debug: Add colored borders to verify both frames are placed
                cv2.rectangle(canvas, (0, 0), (w-1, h-1), (0, 255, 0), 2)  # Green border on left
                cv2.rectangle(canvas, (w+20, 0), (w*2+19, h-1), (0, 0, 255), 2)  # Red border on right
            
            # Center divider
            cv2.line(canvas, (w+10, 0), (w+10, h), (255, 255, 255), 2)
            
            # Resolution info
            info_text = f"Capture: 1920x1080 @ 30fps | Display: {w}x{h} | Frame {self.frame_count}"
            cv2.putText(canvas, info_text, 
                       (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display
            cv2.imshow('1080p Stereo Vision', canvas)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_depth = not show_depth
                print(f"Depth view: {'ON' if show_depth else 'OFF'}")
            elif key == ord('b'):
                show_blur = not show_blur
                print(f"Blur detection: {'ON' if show_blur else 'OFF'}")
            elif key == ord('s'):
                filename = f"stereo_1080p_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, canvas)
                print(f"Saved: {filename}")
            
            self.frame_count += 1
        
        # Cleanup
        self.cap_l.release()
        self.cap_r.release()
        cv2.destroyAllWindows()
        
        if self.fps_history:
            print(f"\nAverage FPS: {np.mean(self.fps_history):.1f}")
        print("Stopped")

if __name__ == "__main__":
    vision = Vision1080p()
    vision.run()