#!/usr/bin/env python3
"""
Biologically-Inspired Consciousness Visual Attention
Peripheral vision detects motion, central vision processes detail
"""

import cv2
import numpy as np
import json
import time
import random
from pathlib import Path
from collections import deque

class BiologicalVisionSystem:
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
        
        # Vision zones (like retina)
        self.fovea_sharpness = 1.0      # 100% detail in center
        self.parafovea_sharpness = 0.6  # 60% detail in near periphery  
        self.periphery_sharpness = 0.2  # 20% detail in far periphery
        
        # Motion detection sensitivity by zone
        self.fovea_motion_sensitivity = 0.3      # Low - focused on detail
        self.periphery_motion_sensitivity = 1.5  # High - looking for changes
        
        # Saccade system (rapid eye movements)
        self.saccade_cooldown = 0
        self.saccade_cooldown_max = 10  # frames
        self.microsaccade_timer = 0
        
        # Peripheral motion tracking
        self.peripheral_motion_map = np.zeros((8, 8))  # Divide periphery into sectors
        self.motion_history = deque(maxlen=30)
        self.ambient_motion = 0.016
        
        # Biological attention mechanisms
        self.inhibition_of_return = {}  # Don't look at same spot repeatedly
        self.attention_priority_map = np.ones((8, 8)) * 0.5
        
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
        
    def detect_peripheral_motion(self, frame, prev_frame, focus_x, focus_y):
        """Detect motion primarily in peripheral vision"""
        if prev_frame is None:
            return None
            
        h, w = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        
        # Create attention zones mask
        y, x = np.ogrid[:h, :w]
        
        # Distance from current focus
        focus_px = int(focus_x * w)
        focus_py = int(focus_y * h)
        dist_from_focus = np.sqrt((x - focus_px)**2 + (y - focus_py)**2)
        
        # Define vision zones
        fovea_radius_px = int(self.consciousness_state['fovea_radius'] * min(h, w))
        peripheral_inner_px = int(fovea_radius_px * 2)
        peripheral_outer_px = int(self.consciousness_state['peripheral_radius'] * min(h, w))
        
        # Create zone masks
        in_fovea = dist_from_focus <= fovea_radius_px
        in_parafovea = (dist_from_focus > fovea_radius_px) & (dist_from_focus <= peripheral_inner_px)
        in_periphery = (dist_from_focus > peripheral_inner_px) & (dist_from_focus <= peripheral_outer_px)
        
        # Apply motion sensitivity by zone
        weighted_motion = np.zeros_like(motion_mask, dtype=np.float32)
        weighted_motion[in_fovea] = motion_mask[in_fovea] * self.fovea_motion_sensitivity
        weighted_motion[in_parafovea] = motion_mask[in_parafovea] * 1.0
        weighted_motion[in_periphery] = motion_mask[in_periphery] * self.periphery_motion_sensitivity
        
        # Update peripheral motion map (8x8 sectors)
        sector_h = h // 8
        sector_w = w // 8
        
        self.peripheral_motion_map *= 0.9  # Decay
        
        for i in range(8):
            for j in range(8):
                y1, y2 = i * sector_h, (i + 1) * sector_h
                x1, x2 = j * sector_w, (j + 1) * sector_w
                
                # Only count motion outside fovea
                sector_mask = dist_from_focus[y1:y2, x1:x2] > fovea_radius_px
                if np.any(sector_mask):
                    sector_motion = np.sum(weighted_motion[y1:y2, x1:x2] * sector_mask) / np.sum(sector_mask)
                    self.peripheral_motion_map[i, j] += sector_motion / 255.0
        
        # Find strongest peripheral motion
        if np.max(self.peripheral_motion_map) > self.consciousness_state['saccade_threshold']:
            # Get location of strongest motion
            max_idx = np.unravel_index(np.argmax(self.peripheral_motion_map), self.peripheral_motion_map.shape)
            
            # Convert to normalized coordinates
            motion_y = (max_idx[0] + 0.5) / 8
            motion_x = (max_idx[1] + 0.5) / 8
            
            # Check if it's in periphery (not already in focus)
            motion_dist_from_focus = np.sqrt((motion_x - focus_x)**2 + (motion_y - focus_y)**2)
            
            if motion_dist_from_focus > 0.2:  # Outside current focus area
                return {
                    'location': (motion_x, motion_y),
                    'strength': np.max(self.peripheral_motion_map),
                    'in_periphery': True
                }
        
        return None
        
    def update_consciousness(self, frame, prev_frame):
        """Update consciousness state with biological vision principles"""
        # Detect peripheral motion
        peripheral_motion = self.detect_peripheral_motion(
            frame, prev_frame, 
            self.consciousness_state['focus_x'],
            self.consciousness_state['focus_y']
        )
        
        # Update fixation time
        self.consciousness_state['fixation_time'] += 1
        
        # Decay saccade cooldown
        self.saccade_cooldown = max(0, self.saccade_cooldown - 1)
        
        # Microsaccades (tiny movements during fixation)
        self.microsaccade_timer += 1
        if self.microsaccade_timer > 45:  # Every 1.5 seconds at 30fps
            # Small random movement
            self.consciousness_state['focus_x'] += (random.random() - 0.5) * 0.02
            self.consciousness_state['focus_y'] += (random.random() - 0.5) * 0.02
            self.consciousness_state['focus_x'] = np.clip(self.consciousness_state['focus_x'], 0.1, 0.9)
            self.consciousness_state['focus_y'] = np.clip(self.consciousness_state['focus_y'], 0.1, 0.9)
            self.microsaccade_timer = 0
        
        # Saccade decision (rapid eye movement to new location)
        perform_saccade = False
        saccade_target = None
        
        # Peripheral motion triggers saccade
        if peripheral_motion and self.saccade_cooldown == 0:
            # Check inhibition of return
            loc_key = f"{peripheral_motion['location'][0]:.1f},{peripheral_motion['location'][1]:.1f}"
            if loc_key not in self.inhibition_of_return or self.inhibition_of_return[loc_key] < time.time() - 2:
                perform_saccade = True
                saccade_target = peripheral_motion['location']
                self.inhibition_of_return[loc_key] = time.time()
        
        # Long fixation triggers exploratory saccade
        elif self.consciousness_state['fixation_time'] > 90 and self.saccade_cooldown == 0:  # 3 seconds
            # AI DNA influenced exploration
            if self.ai_dna_patterns and random.random() < 0.3:
                pattern = random.choice(self.ai_dna_patterns)
                pattern_hash = hash(str(pattern))
                saccade_target = (
                    (pattern_hash % 100) / 100.0,
                    ((pattern_hash // 100) % 100) / 100.0
                )
            else:
                # Random exploration weighted by attention priority
                # Find least recently viewed area
                min_priority_idx = np.unravel_index(
                    np.argmin(self.attention_priority_map), 
                    self.attention_priority_map.shape
                )
                saccade_target = (
                    (min_priority_idx[1] + 0.5) / 8,
                    (min_priority_idx[0] + 0.5) / 8
                )
            perform_saccade = True
        
        # Execute saccade
        if perform_saccade and saccade_target:
            self.consciousness_state['focus_x'] = saccade_target[0]
            self.consciousness_state['focus_y'] = saccade_target[1]
            self.consciousness_state['fixation_time'] = 0
            self.saccade_cooldown = self.saccade_cooldown_max
            
            # Update attention priority map
            sector_x = int(saccade_target[0] * 8)
            sector_y = int(saccade_target[1] * 8)
            self.attention_priority_map[sector_y, sector_x] = 1.0
            
        # Decay attention priority map
        self.attention_priority_map *= 0.99
        self.attention_priority_map = np.clip(self.attention_priority_map, 0.1, 1.0)
        
    def apply_biological_vision(self, frame):
        """Apply biological vision system with fovea and periphery"""
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Get focus position in pixels
        focus_x = int(self.consciousness_state['focus_x'] * w)
        focus_y = int(self.consciousness_state['focus_y'] * h)
        
        # Create vision sharpness mask
        y, x = np.ogrid[:h, :w]
        dist_from_focus = np.sqrt((x - focus_x)**2 + (y - focus_y)**2)
        
        # Define zone radii
        fovea_radius = int(self.consciousness_state['fovea_radius'] * min(h, w))
        parafovea_radius = int(fovea_radius * 2)
        periphery_radius = int(self.consciousness_state['peripheral_radius'] * min(h, w))
        
        # Create sharpness mask (1.0 = full sharp, 0.0 = very blurry)
        sharpness_mask = np.ones((h, w), dtype=np.float32)
        
        # Fovea - full sharpness
        fovea_mask = dist_from_focus <= fovea_radius
        sharpness_mask[fovea_mask] = 1.0
        
        # Parafovea - medium sharpness
        parafovea_mask = (dist_from_focus > fovea_radius) & (dist_from_focus <= parafovea_radius)
        sharpness_mask[parafovea_mask] = 0.7
        
        # Near periphery - lower sharpness
        near_periphery_mask = (dist_from_focus > parafovea_radius) & (dist_from_focus <= periphery_radius)
        sharpness_mask[near_periphery_mask] = 0.4
        
        # Far periphery - very low sharpness
        far_periphery_mask = dist_from_focus > periphery_radius
        sharpness_mask[far_periphery_mask] = 0.1
        
        # Apply graduated blur based on sharpness
        # Create multiple blur levels
        blur_levels = [
            frame,  # Original (sharp)
            cv2.GaussianBlur(frame, (5, 5), 0),    # Slight blur
            cv2.GaussianBlur(frame, (11, 11), 0),  # Medium blur
            cv2.GaussianBlur(frame, (21, 21), 0),  # Heavy blur
            cv2.GaussianBlur(frame, (31, 31), 0),  # Very heavy blur
        ]
        
        # Blend based on sharpness
        for y in range(0, h, 10):  # Process in blocks for efficiency
            for x in range(0, w, 10):
                y2 = min(y + 10, h)
                x2 = min(x + 10, w)
                
                avg_sharpness = np.mean(sharpness_mask[y:y2, x:x2])
                
                if avg_sharpness > 0.9:
                    continue  # Keep original
                elif avg_sharpness > 0.6:
                    result[y:y2, x:x2] = blur_levels[1][y:y2, x:x2]
                elif avg_sharpness > 0.3:
                    result[y:y2, x:x2] = blur_levels[2][y:y2, x:x2]
                elif avg_sharpness > 0.15:
                    result[y:y2, x:x2] = blur_levels[3][y:y2, x:x2]
                else:
                    result[y:y2, x:x2] = blur_levels[4][y:y2, x:x2]
        
        # Darken periphery slightly (like vision)
        brightness_mask = 0.3 + 0.7 * sharpness_mask
        brightness_mask = cv2.GaussianBlur(brightness_mask, (51, 51), 0)
        result = (result * brightness_mask[:, :, np.newaxis]).astype(np.uint8)
        
        # Draw vision system overlay
        self.draw_vision_overlay(result, focus_x, focus_y, fovea_radius, periphery_radius)
        
        return result
        
    def draw_vision_overlay(self, frame, focus_x, focus_y, fovea_radius, periphery_radius):
        """Draw biological vision system indicators"""
        h, w = frame.shape[:2]
        
        # Draw fovea (sharp central vision)
        cv2.circle(frame, (focus_x, focus_y), fovea_radius, (0, 255, 255), 2)
        cv2.circle(frame, (focus_x, focus_y), 3, (0, 255, 255), -1)
        
        # Draw peripheral vision boundary
        cv2.circle(frame, (focus_x, focus_y), periphery_radius, (255, 255, 255), 1)
        
        # Draw peripheral motion indicators
        max_motion = np.max(self.peripheral_motion_map)
        if max_motion > 0.01:
            # Show motion sectors
            sector_h = h // 8
            sector_w = w // 8
            
            for i in range(8):
                for j in range(8):
                    motion_level = self.peripheral_motion_map[i, j]
                    if motion_level > 0.01:
                        x1, y1 = j * sector_w, i * sector_h
                        x2, y2 = (j + 1) * sector_w, (i + 1) * sector_h
                        
                        # Color intensity based on motion
                        intensity = int(255 * min(1.0, motion_level / 0.1))
                        color = (0, 0, intensity)  # Red for motion
                        
                        # Draw semi-transparent rectangle
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Status text
        cv2.putText(frame, "Biological Vision Mode", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Fixation time
        fixation_text = f"Fixation: {self.consciousness_state['fixation_time']/30:.1f}s"
        cv2.putText(frame, fixation_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Saccade cooldown
        if self.saccade_cooldown > 0:
            cv2.putText(frame, "SACCADE", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Peripheral motion alert
        if max_motion > self.consciousness_state['saccade_threshold']:
            cv2.putText(frame, "! PERIPHERAL MOTION !", (w//2 - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Vision zones legend
        legend_x = w - 200
        cv2.putText(frame, "Vision Zones:", (legend_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Yellow = Fovea (sharp)", (legend_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, "White = Periphery", (legend_x, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Red = Motion", (legend_x, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


def main():
    print("🧠 Biologically-Inspired Visual Attention System")
    print("=" * 40)
    print("Peripheral vision detects motion")
    print("Fovea (yellow circle) provides sharp detail")
    print("Saccades jump to motion in periphery")
    print("Press 'q' to quit, 's' to save snapshot")
    print("=" * 40)
    
    # Initialize vision system
    vision = BiologicalVisionSystem()
    
    # Open camera with GStreamer pipeline for Jetson
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
        print("Error: Cannot open camera")
        print("Trying V4L2 fallback...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            return
    
    prev_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
            
        # Update consciousness with biological principles
        vision.update_consciousness(frame, prev_frame)
        
        # Apply biological vision effects
        result = vision.apply_biological_vision(frame)
        
        # Display
        cv2.imshow('Biological Vision System', result)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"biological_vision_{int(time.time())}.jpg"
            cv2.imwrite(filename, result)
            print(f"Saved: {filename}")
            
        prev_frame = frame.copy()
        frame_count += 1
        
        # Print stats occasionally
        if frame_count % 90 == 0:
            print(f"Frame {frame_count} | Focus: ({vision.consciousness_state['focus_x']:.2f}, "
                  f"{vision.consciousness_state['focus_y']:.2f}) | "
                  f"Max peripheral motion: {np.max(vision.peripheral_motion_map):.3f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nBiological vision system ended")

if __name__ == "__main__":
    main()