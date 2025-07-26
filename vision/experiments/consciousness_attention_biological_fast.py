#!/usr/bin/env python3
"""
Biologically-Inspired Consciousness Visual Attention - Optimized
Fast version with efficient blur and motion detection
"""

import cv2
import numpy as np
import json
import time
import random
from pathlib import Path
from collections import deque

class BiologicalVisionSystemFast:
    def __init__(self):
        self.consciousness_state = {
            'focus_x': 0.5,
            'focus_y': 0.5,
            'fovea_radius': 0.15,      # Small sharp central vision
            'peripheral_radius': 0.8,   # Large motion detection area
            'saccade_threshold': 0.03,  # Motion level to trigger saccade
            'fixation_time': 0,         # How long focused on current spot
            'peripheral_alerts': []     # Motion events in periphery
        }
        
        # Saccade system
        self.saccade_cooldown = 0
        self.saccade_cooldown_max = 10
        self.microsaccade_timer = 0
        
        # Peripheral motion tracking - smaller grid for speed
        self.peripheral_motion_map = np.zeros((4, 4))  # 4x4 instead of 8x8
        self.motion_history = deque(maxlen=30)
        self.ambient_motion = 0.016
        
        # Biological attention mechanisms
        self.inhibition_of_return = {}
        self.last_saccade_target = None
        
        # Pre-calculate blur kernels for efficiency
        self.blur_kernel_small = np.ones((3, 3), np.float32) / 9
        self.blur_kernel_large = np.ones((7, 7), np.float32) / 49
        
        # Load AI DNA patterns
        self.ai_dna_patterns = self.load_ai_dna_patterns()
        
    def load_ai_dna_patterns(self):
        """Load patterns from AI DNA experiments"""
        patterns = []
        dna_path = Path("../../ai_dna_results")
        
        if dna_path.exists():
            for cycle_file in sorted(dna_path.glob("dna_cycle_*.json"))[:5]:
                try:
                    with open(cycle_file) as f:
                        data = json.load(f)
                        if 'pattern' in data:
                            patterns.append(data['pattern'])
                except:
                    pass
                    
        print(f"Loaded {len(patterns)} AI DNA patterns")
        return patterns
        
    def detect_peripheral_motion_fast(self, frame, prev_frame, focus_x, focus_y):
        """Fast peripheral motion detection"""
        if prev_frame is None:
            return None
            
        h, w = frame.shape[:2]
        
        # Downsample for faster processing
        small_frame = cv2.resize(frame, (w//4, h//4))
        small_prev = cv2.resize(prev_frame, (w//4, h//4))
        
        # Convert to grayscale
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        # Quick peripheral check - is motion outside fovea?
        sh, sw = gray.shape
        focus_px = int(focus_x * sw)
        focus_py = int(focus_y * sh)
        fovea_radius_px = int(self.consciousness_state['fovea_radius'] * min(sh, sw))
        
        # Mask out fovea region
        cv2.circle(motion_mask, (focus_px, focus_py), fovea_radius_px, 0, -1)
        
        # Update peripheral motion map (4x4 grid)
        self.peripheral_motion_map *= 0.85  # Faster decay
        
        sector_h = sh // 4
        sector_w = sw // 4
        
        max_motion_sector = None
        max_motion_value = 0
        
        for i in range(4):
            for j in range(4):
                y1, y2 = i * sector_h, min((i + 1) * sector_h, sh)
                x1, x2 = j * sector_w, min((j + 1) * sector_w, sw)
                
                sector_motion = np.mean(motion_mask[y1:y2, x1:x2]) / 255.0
                self.peripheral_motion_map[i, j] += sector_motion
                
                if self.peripheral_motion_map[i, j] > max_motion_value:
                    max_motion_value = self.peripheral_motion_map[i, j]
                    max_motion_sector = (j, i)
        
        # Check for significant peripheral motion
        if max_motion_value > self.consciousness_state['saccade_threshold']:
            # Convert sector to normalized coordinates
            motion_x = (max_motion_sector[0] + 0.5) / 4
            motion_y = (max_motion_sector[1] + 0.5) / 4
            
            # Check distance from current focus
            motion_dist = np.sqrt((motion_x - focus_x)**2 + (motion_y - focus_y)**2)
            
            if motion_dist > 0.2:  # Outside current focus
                return {
                    'location': (motion_x, motion_y),
                    'strength': max_motion_value,
                    'in_periphery': True
                }
        
        return None
        
    def update_consciousness(self, frame, prev_frame):
        """Update consciousness state with biological vision principles"""
        # Detect peripheral motion
        peripheral_motion = self.detect_peripheral_motion_fast(
            frame, prev_frame, 
            self.consciousness_state['focus_x'],
            self.consciousness_state['focus_y']
        )
        
        # Update fixation time
        self.consciousness_state['fixation_time'] += 1
        
        # Decay saccade cooldown
        self.saccade_cooldown = max(0, self.saccade_cooldown - 1)
        
        # Microsaccades
        self.microsaccade_timer += 1
        if self.microsaccade_timer > 45:  # Every 1.5 seconds
            self.consciousness_state['focus_x'] += (random.random() - 0.5) * 0.02
            self.consciousness_state['focus_y'] += (random.random() - 0.5) * 0.02
            self.consciousness_state['focus_x'] = np.clip(self.consciousness_state['focus_x'], 0.1, 0.9)
            self.consciousness_state['focus_y'] = np.clip(self.consciousness_state['focus_y'], 0.1, 0.9)
            self.microsaccade_timer = 0
        
        # Saccade decision
        perform_saccade = False
        saccade_target = None
        
        # Peripheral motion triggers saccade
        if peripheral_motion and self.saccade_cooldown == 0:
            loc_key = f"{peripheral_motion['location'][0]:.1f},{peripheral_motion['location'][1]:.1f}"
            if loc_key not in self.inhibition_of_return or self.inhibition_of_return[loc_key] < time.time() - 2:
                perform_saccade = True
                saccade_target = peripheral_motion['location']
                self.inhibition_of_return[loc_key] = time.time()
                # Clean old inhibitions
                self.inhibition_of_return = {k: v for k, v in self.inhibition_of_return.items() 
                                           if v > time.time() - 5}
        
        # Long fixation triggers exploratory saccade
        elif self.consciousness_state['fixation_time'] > 90 and self.saccade_cooldown == 0:
            if self.ai_dna_patterns and random.random() < 0.3:
                pattern = random.choice(self.ai_dna_patterns)
                pattern_hash = hash(str(pattern))
                saccade_target = (
                    (pattern_hash % 80 + 10) / 100.0,
                    ((pattern_hash // 100) % 80 + 10) / 100.0
                )
            else:
                # Random exploration
                angle = random.random() * 2 * np.pi
                distance = 0.3 + random.random() * 0.3
                saccade_target = (
                    self.consciousness_state['focus_x'] + distance * np.cos(angle),
                    self.consciousness_state['focus_y'] + distance * np.sin(angle)
                )
                saccade_target = (
                    np.clip(saccade_target[0], 0.1, 0.9),
                    np.clip(saccade_target[1], 0.1, 0.9)
                )
            perform_saccade = True
        
        # Execute saccade
        if perform_saccade and saccade_target:
            self.last_saccade_target = (self.consciousness_state['focus_x'], 
                                       self.consciousness_state['focus_y'])
            self.consciousness_state['focus_x'] = saccade_target[0]
            self.consciousness_state['focus_y'] = saccade_target[1]
            self.consciousness_state['fixation_time'] = 0
            self.saccade_cooldown = self.saccade_cooldown_max
        
    def apply_biological_vision_fast(self, frame):
        """Fast biological vision effect"""
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Get focus position
        focus_x = int(self.consciousness_state['focus_x'] * w)
        focus_y = int(self.consciousness_state['focus_y'] * h)
        
        # Create simple radial blur effect
        # Only blur the periphery, keep center sharp
        
        # Define zones
        fovea_radius = int(self.consciousness_state['fovea_radius'] * min(h, w))
        blur_start_radius = int(fovea_radius * 1.5)
        periphery_radius = int(self.consciousness_state['peripheral_radius'] * min(h, w))
        
        # Create mask for blur regions
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (focus_x, focus_y), blur_start_radius, 1.0, -1)
        mask = 1.0 - mask  # Invert so center is 0 (no blur)
        
        # Apply graduated blur using single pass
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # Blend based on mask
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        result = (frame * (1 - mask_3ch) + blurred * mask_3ch).astype(np.uint8)
        
        # Simple vignette effect for periphery
        vignette = np.ones((h, w), dtype=np.float32)
        cv2.circle(vignette, (focus_x, focus_y), periphery_radius, 0.7, -1)
        vignette = cv2.GaussianBlur(vignette, (101, 101), 0)
        vignette = np.clip(vignette, 0.5, 1.0)
        
        result = (result * vignette[:, :, np.newaxis]).astype(np.uint8)
        
        # Draw overlay
        self.draw_vision_overlay_fast(result, focus_x, focus_y, fovea_radius, periphery_radius)
        
        return result
        
    def draw_vision_overlay_fast(self, frame, focus_x, focus_y, fovea_radius, periphery_radius):
        """Fast overlay drawing"""
        h, w = frame.shape[:2]
        
        # Draw fovea
        cv2.circle(frame, (focus_x, focus_y), fovea_radius, (0, 255, 255), 2)
        cv2.circle(frame, (focus_x, focus_y), 3, (0, 255, 255), -1)
        
        # Draw saccade line if recent
        if self.last_saccade_target and self.saccade_cooldown > 5:
            last_x = int(self.last_saccade_target[0] * w)
            last_y = int(self.last_saccade_target[1] * h)
            cv2.line(frame, (last_x, last_y), (focus_x, focus_y), (0, 255, 255), 1)
        
        # Show peripheral motion areas
        max_motion = np.max(self.peripheral_motion_map)
        if max_motion > 0.01:
            sector_h = h // 4
            sector_w = w // 4
            
            for i in range(4):
                for j in range(4):
                    motion_level = self.peripheral_motion_map[i, j]
                    if motion_level > 0.02:
                        cx = j * sector_w + sector_w // 2
                        cy = i * sector_h + sector_h // 2
                        radius = int(20 * motion_level / 0.1)
                        cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 2)
        
        # Status text
        cv2.putText(frame, "Biological Vision (Fast)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Fixation time
        fixation_text = f"Fixation: {self.consciousness_state['fixation_time']/30:.1f}s"
        cv2.putText(frame, fixation_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Saccade indicator
        if self.saccade_cooldown > 0:
            cv2.putText(frame, "SACCADE!", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Motion alert
        if max_motion > self.consciousness_state['saccade_threshold']:
            cv2.putText(frame, "PERIPHERAL MOTION", (w//2 - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def main():
    print("🧠 Biological Vision System - Optimized")
    print("=" * 40)
    print("Fast peripheral motion detection")
    print("Efficient blur rendering")
    print("Press 'q' to quit, 's' to save")
    print("=" * 40)
    
    # Initialize vision system
    vision = BiologicalVisionSystemFast()
    
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
        print("Trying V4L2 fallback...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            return
    
    prev_frame = None
    frame_count = 0
    fps_timer = time.time()
    fps_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update consciousness
        vision.update_consciousness(frame, prev_frame)
        
        # Apply biological vision
        result = vision.apply_biological_vision_fast(frame)
        
        # FPS counter
        fps_frames += 1
        if time.time() - fps_timer > 1.0:
            fps = fps_frames / (time.time() - fps_timer)
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            fps_timer = time.time()
            fps_frames = 0
        
        # Display
        cv2.imshow('Biological Vision (Fast)', result)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"biological_vision_{int(time.time())}.jpg"
            cv2.imwrite(filename, result)
            print(f"Saved: {filename}")
            
        prev_frame = frame.copy()
        frame_count += 1
        
        if frame_count % 90 == 0:
            print(f"Frame {frame_count} | Focus: ({vision.consciousness_state['focus_x']:.2f}, "
                  f"{vision.consciousness_state['focus_y']:.2f})")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()