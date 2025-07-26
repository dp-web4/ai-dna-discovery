#!/usr/bin/env python3
"""
Clean Consciousness Visual Attention
Focus area for processing, periphery for motion detection only
No blur - just functional separation
"""

import cv2
import numpy as np
import time
import random
from collections import deque

class CleanConsciousnessVision:
    def __init__(self):
        # Focus area properties
        self.focus_x = 0.5
        self.focus_y = 0.5
        self.focus_radius = 0.15  # Can dilate/shrink
        self.focus_radius_target = 0.15
        
        # Motion detection
        self.motion_grid_size = 8  # 8x8 grid for periphery
        self.motion_heatmap = np.zeros((self.motion_grid_size, self.motion_grid_size))
        self.ambient_motion = 0.016
        self.motion_history = deque(maxlen=30)
        
        # Saccade control
        self.fixation_timer = 0
        self.saccade_cooldown = 0
        self.last_focus = (0.5, 0.5)
        
    def process_frame(self, frame, prev_frame):
        """Main processing - split into focus and periphery"""
        h, w = frame.shape[:2]
        
        # Define focus area
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        focus_radius_px = int(self.focus_radius * min(h, w))
        
        # Create masks
        focus_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(focus_mask, (focus_px, focus_py), focus_radius_px, 255, -1)
        periphery_mask = 255 - focus_mask
        
        # Process periphery for motion only
        motion_peaks = []
        if prev_frame is not None:
            motion_peaks = self.detect_peripheral_motion(frame, prev_frame, periphery_mask)
        
        # Update focus based on motion
        self.update_focus(motion_peaks)
        
        # Smooth radius changes
        self.focus_radius += (self.focus_radius_target - self.focus_radius) * 0.1
        
        # Create visualization
        result = self.visualize(frame, focus_mask, motion_peaks)
        
        return result
        
    def detect_peripheral_motion(self, frame, prev_frame, periphery_mask):
        """Detect motion only in peripheral areas"""
        h, w = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion
        diff = cv2.absdiff(gray, prev_gray)
        
        # Apply periphery mask - ignore motion in focus area
        diff = cv2.bitwise_and(diff, diff, mask=periphery_mask)
        
        # Update motion heatmap
        self.motion_heatmap *= 0.9  # Decay
        
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        # Calculate motion for each grid cell
        current_motion_grid = np.zeros((self.motion_grid_size, self.motion_grid_size))
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                y1 = i * grid_h
                y2 = min((i + 1) * grid_h, h)
                x1 = j * grid_w
                x2 = min((j + 1) * grid_w, w)
                
                # Get motion in this cell
                cell_diff = diff[y1:y2, x1:x2]
                cell_motion = np.mean(cell_diff) / 255.0
                
                current_motion_grid[i, j] = cell_motion
                self.motion_heatmap[i, j] += cell_motion
        
        # Calculate average motion (for ambient baseline)
        avg_motion = np.mean(current_motion_grid[current_motion_grid > 0]) if np.any(current_motion_grid > 0) else self.ambient_motion
        self.motion_history.append(avg_motion)
        
        # Find peaks above threshold (P/A > 1.5)
        motion_peaks = []
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                cell_motion = self.motion_heatmap[i, j]
                pa_ratio = cell_motion / self.ambient_motion
                
                if pa_ratio > 1.5:
                    # Convert grid position to normalized coordinates
                    peak_x = (j + 0.5) / self.motion_grid_size
                    peak_y = (i + 0.5) / self.motion_grid_size
                    
                    motion_peaks.append({
                        'x': peak_x,
                        'y': peak_y,
                        'pa_ratio': pa_ratio,
                        'strength': cell_motion
                    })
        
        # Sort by P/A ratio (highest first)
        motion_peaks.sort(key=lambda p: p['pa_ratio'], reverse=True)
        
        return motion_peaks
        
    def update_focus(self, motion_peaks):
        """Update focus position and size based on motion"""
        self.fixation_timer += 1
        self.saccade_cooldown = max(0, self.saccade_cooldown - 1)
        
        # Check for saccade triggers
        perform_saccade = False
        target = None
        
        # High motion in periphery triggers saccade
        if motion_peaks and self.saccade_cooldown == 0:
            # Take highest P/A ratio peak
            peak = motion_peaks[0]
            if peak['pa_ratio'] > 2.0:  # Strong motion
                target = (peak['x'], peak['y'])
                perform_saccade = True
                # Dilate focus for new area
                self.focus_radius_target = 0.2
        
        # Long fixation triggers exploration
        elif self.fixation_timer > 90 and self.saccade_cooldown == 0:  # 3 seconds
            # Random jump
            angle = random.random() * 2 * np.pi
            distance = 0.3 + random.random() * 0.2
            target = (
                self.focus_x + distance * np.cos(angle),
                self.focus_y + distance * np.sin(angle)
            )
            target = (
                np.clip(target[0], 0.1, 0.9),
                np.clip(target[1], 0.1, 0.9)
            )
            perform_saccade = True
            # Shrink focus for exploration
            self.focus_radius_target = 0.12
        
        # Execute saccade
        if perform_saccade and target:
            self.last_focus = (self.focus_x, self.focus_y)
            self.focus_x = target[0]
            self.focus_y = target[1]
            self.fixation_timer = 0
            self.saccade_cooldown = 10
        
        # Gradual radius adjustment based on motion
        if motion_peaks:
            # More motion = larger focus
            avg_pa = np.mean([p['pa_ratio'] for p in motion_peaks[:3]])
            if avg_pa > 3.0:
                self.focus_radius_target = 0.25
            elif avg_pa > 2.0:
                self.focus_radius_target = 0.18
            else:
                self.focus_radius_target = 0.15
        else:
            # No motion = shrink slightly
            self.focus_radius_target = max(0.12, self.focus_radius_target * 0.98)
            
    def visualize(self, frame, focus_mask, motion_peaks):
        """Create visualization"""
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Darken periphery slightly (not blur, just dimmer)
        periphery = cv2.bitwise_and(result, result, mask=255-focus_mask)
        periphery = (periphery * 0.7).astype(np.uint8)
        focus = cv2.bitwise_and(result, result, mask=focus_mask)
        result = cv2.add(periphery, focus)
        
        # Draw focus circle
        focus_px = int(self.focus_x * w)
        focus_py = int(self.focus_y * h)
        focus_radius_px = int(self.focus_radius * min(h, w))
        cv2.circle(result, (focus_px, focus_py), focus_radius_px, (0, 255, 255), 2)
        cv2.circle(result, (focus_px, focus_py), 3, (0, 255, 255), -1)
        
        # Draw motion heatmap overlay
        grid_h = h // self.motion_grid_size
        grid_w = w // self.motion_grid_size
        
        for i in range(self.motion_grid_size):
            for j in range(self.motion_grid_size):
                if self.motion_heatmap[i, j] > 0.01:
                    x1 = j * grid_w
                    y1 = i * grid_h
                    x2 = (j + 1) * grid_w
                    y2 = (i + 1) * grid_h
                    
                    # Heat intensity
                    intensity = min(1.0, self.motion_heatmap[i, j] / 0.05)
                    color = (0, int(100 * intensity), int(255 * intensity))  # Blue to red
                    
                    # Draw semi-transparent rectangle
                    overlay = result.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
        
        # Draw motion peaks
        for i, peak in enumerate(motion_peaks[:5]):  # Top 5 peaks
            px = int(peak['x'] * w)
            py = int(peak['y'] * h)
            radius = int(10 + peak['pa_ratio'] * 5)
            cv2.circle(result, (px, py), radius, (0, 0, 255), 2)
            cv2.putText(result, f"{peak['pa_ratio']:.1f}", (px-15, py-radius-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw saccade line if recent
        if self.saccade_cooldown > 5:
            last_px = int(self.last_focus[0] * w)
            last_py = int(self.last_focus[1] * h)
            cv2.arrowedLine(result, (last_px, last_py), (focus_px, focus_py), 
                           (0, 255, 255), 2, tipLength=0.1)
        
        # Status text
        cv2.putText(result, "Focus/Periphery Separation", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Focus info
        focus_text = f"Focus: ({self.focus_x:.2f}, {self.focus_y:.2f}) R: {self.focus_radius:.2f}"
        cv2.putText(result, focus_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Motion info
        if self.motion_history:
            ambient = np.mean(self.motion_history)
            motion_text = f"Ambient: {ambient:.3f}"
            cv2.putText(result, motion_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Peak count
        if motion_peaks:
            peaks_text = f"Peaks (P/A>1.5): {len(motion_peaks)}"
            cv2.putText(result, peaks_text, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Saccade indicator
        if self.saccade_cooldown > 0:
            cv2.putText(result, "SACCADE", (w//2 - 50, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return result

def main():
    print("👁️ Clean Focus/Periphery Vision System")
    print("=" * 40)
    print("Focus area: Detailed processing (yellow circle)")
    print("Periphery: Motion detection only (heat map)")
    print("Red circles: Motion peaks with P/A ratios")
    print("Focus jumps to highest P/A > 2.0")
    print("Press 'q' to quit, 's' to save")
    print("=" * 40)
    
    vision = CleanConsciousnessVision()
    
    # Open camera
    gst_pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("Could not open camera")
            return
    
    prev_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = vision.process_frame(frame, prev_frame)
        
        # Display
        cv2.imshow('Clean Consciousness Vision', result)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"clean_vision_{int(time.time())}.jpg"
            cv2.imwrite(filename, result)
            print(f"Saved: {filename}")
        
        prev_frame = frame.copy()
        frame_count += 1
        
        if frame_count % 90 == 0:
            if vision.motion_history:
                avg_ambient = np.mean(vision.motion_history)
                print(f"Frame {frame_count} | Focus: ({vision.focus_x:.2f}, {vision.focus_y:.2f}) "
                      f"| Radius: {vision.focus_radius:.2f} | Ambient: {avg_ambient:.3f}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()