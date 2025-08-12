#!/usr/bin/env python3
"""
Direct test of dashboard with camera feeds
August 12, 2025
"""

import cv2
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def gst_pipeline(sensor_id=0):
    """Create GStreamer pipeline for CSI camera"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=2 ! "
        f"video/x-raw(memory:NVMM), width=1920, height=1080, "
        f"format=NV12, framerate=30/1 ! "
        f"nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

def main():
    print("Testing Dashboard with Direct Camera Access")
    print("="*50)
    
    # Initialize cameras
    print("Initializing cameras...")
    cap_l = cv2.VideoCapture(gst_pipeline(0), cv2.CAP_GSTREAMER)
    cap_r = cv2.VideoCapture(gst_pipeline(1), cv2.CAP_GSTREAMER)
    
    # Create window
    window_name = "Coherence Dashboard Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)
    
    # Test variables
    reality_field = 0.5
    context = "STABLE"
    tick = 0
    
    print("Dashboard window created. Press 'q' to quit")
    
    while True:
        # Create dashboard canvas
        dashboard = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)  # Dark gray
        
        # Read camera frames
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        
        # Draw camera feeds at top
        if ret_l and frame_l is not None:
            h, w = frame_l.shape[:2]
            # Ensure frame fits in dashboard
            end_x = min(20 + w, 950)  # Leave space between cameras
            dashboard[20:20+h, 20:end_x] = frame_l[:, :end_x-20]
            cv2.putText(dashboard, "Left Camera", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if ret_r and frame_r is not None:
            h, w = frame_r.shape[:2]
            # Place right camera with proper spacing
            start_x = 970
            end_x = min(start_x + w, 1920)
            actual_width = end_x - start_x
            dashboard[20:20+h, start_x:end_x] = frame_r[:, :actual_width]
            cv2.putText(dashboard, "Right Camera", (start_x + 10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw reality field visualization (center)
        center_x, center_y = 960, 700
        radius = int(100 * (1 + reality_field))
        color_intensity = int(255 * reality_field)
        cv2.circle(dashboard, (center_x, center_y), radius,
                  (0, color_intensity, 255-color_intensity), -1)
        cv2.circle(dashboard, (center_x, center_y), radius,
                  (255, 255, 255), 2)
        
        # Add text overlays
        cv2.putText(dashboard, f"Reality Field: {reality_field:.3f}",
                   (center_x - 120, center_y - 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(dashboard, f"Context: {context}",
                   (center_x - 80, center_y + 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(dashboard, f"Tick: {tick}",
                   (20, 1050),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Display the dashboard
        cv2.imshow(window_name, dashboard)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"dashboard_test_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, dashboard)
            print(f"Screenshot saved: {filename}")
        
        # Update simulation
        tick += 1
        reality_field = 0.5 + 0.3 * np.sin(tick * 0.05)
        
        # Change context periodically
        if tick % 100 == 0:
            contexts = ["STABLE", "MOVING", "UNSTABLE", "NOVEL"]
            context = contexts[(tick // 100) % 4]
        
        time.sleep(0.033)  # ~30 FPS
    
    # Cleanup
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print("Dashboard test complete")

if __name__ == "__main__":
    main()