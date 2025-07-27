#!/usr/bin/env python3
"""
Debug version of binocular consciousness with visible motion heatmap
"""

import cv2
import numpy as np
import time
from binocular_consciousness import IndependentEye, EyeConfig, StereoCorrelationEngine

class DebugEye(IndependentEye):
    """Eye with debug visualization"""
    
    def _visualize(self, frame, h, w):
        """Enhanced visualization with motion heatmap"""
        result = frame.copy()
        
        # Draw motion heatmap overlay
        heatmap_vis = np.zeros((h, w, 3), dtype=np.uint8)
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        max_motion = np.max(self.motion_heatmap) if np.max(self.motion_heatmap) > 0 else 1.0
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                y1, y2 = i * grid_h, min((i + 1) * grid_h, h)
                x1, x2 = j * grid_w, min((j + 1) * grid_w, w)
                
                # Normalized motion value
                motion_val = self.motion_heatmap[i, j] / max_motion if max_motion > 0 else 0
                
                if motion_val > 0.01:
                    # Heat color based on motion
                    color = (0, int(100 * motion_val), int(255 * motion_val))
                    cv2.rectangle(heatmap_vis, (x1, y1), (x2, y2), color, -1)
                    
                    # Show value
                    if motion_val > 0.1:
                        cv2.putText(heatmap_vis, f"{motion_val:.2f}", (x1+5, y1+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Blend heatmap with frame
        result = cv2.addWeighted(result, 0.7, heatmap_vis, 0.3, 0)
        
        # Draw focus circle
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        radius_px = int(self.focus_radius * min(h, w))
        
        # Eye-specific color
        color = (255, 100, 0) if self.config.eye_id == "left" else (0, 100, 255)
        cv2.circle(result, (focus_px, focus_py), radius_px, color, 3)
        
        # Draw crosshair at focus point
        cv2.line(result, (focus_px - 10, focus_py), (focus_px + 10, focus_py), color, 2)
        cv2.line(result, (focus_px, focus_py - 10), (focus_px, focus_py + 10), color, 2)
        
        # Labels
        cv2.putText(result, f"{self.config.eye_id.upper()} EYE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Debug info
        cv2.putText(result, f"Focus: ({self.focus_x:.2f}, {self.focus_y:.2f})", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Max motion: {np.max(self.motion_heatmap):.3f}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Ambient: {self.ambient_motion:.3f}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result

def main():
    print("🧠 Debug Binocular Consciousness")
    print("="*40)
    print("Motion heatmap visible as blue overlay")
    print("Values shown in grid cells")
    print("Focus should move to high motion areas")
    print("\nPress 'q' to quit")
    print("Press 'm' to trigger manual motion")
    
    # Create debug eyes
    left_eye = DebugEye(EyeConfig(
        eye_id="left",
        sensor_id=0,
        position_offset=-1.5,
        rotation_offset=0.0
    ))
    
    right_eye = DebugEye(EyeConfig(
        eye_id="right",
        sensor_id=1,
        position_offset=1.5,
        rotation_offset=0.0
    ))
    
    # Camera setup
    def get_pipeline(sensor_id):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1"
        )
    
    cap0 = cv2.VideoCapture(get_pipeline(0), cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(get_pipeline(1), cv2.CAP_GSTREAMER)
    
    if not cap0.isOpened() or not cap1.isOpened():
        print("Failed to open cameras")
        return
        
    # Manual motion trigger
    trigger_motion = False
    motion_x = 0.8
    motion_y = 0.2
    
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if ret0 and ret1:
            # Add artificial motion if triggered
            if trigger_motion:
                h, w = frame0.shape[:2]
                mx = int(motion_x * w)
                my = int(motion_y * h)
                cv2.circle(frame0, (mx, my), 30, (255, 255, 255), -1)
                cv2.circle(frame1, (mx, my), 30, (255, 255, 255), -1)
                trigger_motion = False
            
            # Process frames
            left_vis = left_eye.process_frame(frame0)
            right_vis = right_eye.process_frame(frame1)
            
            # Combine
            stereo = np.hstack([left_vis, right_vis])
            
            # Status
            cv2.putText(stereo, "Motion Detection Debug - Press 'm' to trigger motion", 
                       (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Binocular Debug', stereo)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                trigger_motion = True
                # Random position
                motion_x = np.random.uniform(0.2, 0.8)
                motion_y = np.random.uniform(0.2, 0.8)
                print(f"Triggering motion at ({motion_x:.2f}, {motion_y:.2f})")
                
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()