#!/usr/bin/env python3
"""
Consciousness-Guided Visual Attention - Enhanced Motion Response
Peak motion tracking with proportional focus response
"""

import cv2
import numpy as np
import json
import time
import random
from pathlib import Path
from collections import deque

class ConsciousnessAttention:
    def __init__(self):
        self.consciousness_state = {
            'focus_x': 0.5,
            'focus_y': 0.5,
            'attention_radius': 0.3,
            'curiosity': 0.5,
            'pattern_memory': []
        }
        
        # Motion detection parameters
        self.motion_threshold = 5
        self.motion_history = deque(maxlen=60)  # 2 seconds at 30fps
        
        # Peak motion tracking
        self.peak_motion_region = {'x': 0.5, 'y': 0.5, 'strength': 0.0}
        self.peak_decay = 0.95  # How fast peak decays
        self.ambient_motion = 0.016  # Calibrated ambient
        self.motion_regions = {}  # Track motion heat map
        
        # Motion response parameters
        self.sustained_motion_frames = 0
        self.motion_persistence_threshold = 5  # frames
        
        # Load AI DNA patterns if available
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
        
    def update_peak_motion(self, motion_center, motion_strength):
        """Update peak motion region with decay"""
        # Decay existing peak
        self.peak_motion_region['strength'] *= self.peak_decay
        
        # Update if new motion is stronger
        if motion_strength > self.peak_motion_region['strength']:
            self.peak_motion_region['x'] = motion_center[0]
            self.peak_motion_region['y'] = motion_center[1]
            self.peak_motion_region['strength'] = motion_strength
        
    def update_consciousness(self, frame_stats):
        """Update consciousness state based on visual input"""
        motion = frame_stats.get('motion', 0)
        motion_center = frame_stats.get('motion_center', (0.5, 0.5))
        
        # Store motion history
        self.motion_history.append(motion)
        
        # Calculate motion dynamics
        avg_motion = np.mean(self.motion_history) if self.motion_history else self.ambient_motion
        peak_motion = max(self.motion_history) if self.motion_history else self.ambient_motion
        
        # Peak to ambient ratio (P/A ratio)
        pa_ratio = peak_motion / max(self.ambient_motion, 0.001)
        
        # Motion delta from ambient
        motion_delta = max(0, motion - self.ambient_motion)
        
        # Update peak motion tracking
        self.update_peak_motion(motion_center, motion_delta)
        
        # Sustained motion detection
        if motion > self.ambient_motion * 1.5:
            self.sustained_motion_frames += 1
        else:
            self.sustained_motion_frames = max(0, self.sustained_motion_frames - 2)
        
        # Curiosity based on P/A ratio and sustained motion
        # High P/A ratio (>3) = high curiosity, Low P/A (<1.5) = low curiosity
        curiosity_target = min(1.0, (pa_ratio - 1.0) / 3.0)
        curiosity_target = max(0.1, curiosity_target)  # Minimum curiosity
        
        # Add boost for sustained motion
        if self.sustained_motion_frames > self.motion_persistence_threshold:
            curiosity_target = min(1.0, curiosity_target + 0.3)
        
        # Smooth curiosity transitions
        self.consciousness_state['curiosity'] = (
            self.consciousness_state['curiosity'] * 0.7 + curiosity_target * 0.3)
        
        # Focus response based on P/A ratio
        # High P/A = strong snap to motion, Low P/A = gentle drift
        if pa_ratio > 3.0 and motion > self.ambient_motion * 1.5:
            # Strong motion - snap to peak motion region
            snap_strength = min(1.0, (pa_ratio - 2.0) / 5.0)  # 0 at PA=2, 1 at PA=7+
            
            # Use peak motion region for sustained motion
            if self.peak_motion_region['strength'] > 0.01:
                target_x = self.peak_motion_region['x']
                target_y = self.peak_motion_region['y']
            else:
                target_x = motion_center[0]
                target_y = motion_center[1]
            
            self.consciousness_state['focus_x'] = (
                self.consciousness_state['focus_x'] * (1 - snap_strength) + 
                target_x * snap_strength)
            self.consciousness_state['focus_y'] = (
                self.consciousness_state['focus_y'] * (1 - snap_strength) + 
                target_y * snap_strength)
            
        elif pa_ratio > 1.5:
            # Moderate motion - drift toward motion
            drift_strength = (pa_ratio - 1.5) / 1.5  # 0 at PA=1.5, 1 at PA=3
            drift_strength *= 0.3  # Max 30% drift
            
            self.consciousness_state['focus_x'] = (
                self.consciousness_state['focus_x'] * (1 - drift_strength) + 
                motion_center[0] * drift_strength)
            self.consciousness_state['focus_y'] = (
                self.consciousness_state['focus_y'] * (1 - drift_strength) + 
                motion_center[1] * drift_strength)
            
        else:
            # Low motion - wander freely
            wander_scale = 0.02 * (1 - self.consciousness_state['curiosity'])
            
            # AI DNA influence during low motion
            if self.ai_dna_patterns and random.random() < 0.1:
                pattern = random.choice(self.ai_dna_patterns)
                pattern_hash = hash(str(pattern))
                self.consciousness_state['focus_x'] = (pattern_hash % 100) / 100.0
                self.consciousness_state['focus_y'] = ((pattern_hash // 100) % 100) / 100.0
            else:
                # Random drift
                drift_x = (random.random() - 0.5) * wander_scale
                drift_y = (random.random() - 0.5) * wander_scale
                
                self.consciousness_state['focus_x'] = np.clip(
                    self.consciousness_state['focus_x'] + drift_x, 0.1, 0.9)
                self.consciousness_state['focus_y'] = np.clip(
                    self.consciousness_state['focus_y'] + drift_y, 0.1, 0.9)
        
        # Attention radius based on curiosity and motion
        pulse = np.sin(time.time() * 2) * 0.03
        base_radius = 0.2 + (self.consciousness_state['curiosity'] * 0.35)
        
        # Expand radius for sustained motion
        if self.sustained_motion_frames > self.motion_persistence_threshold:
            base_radius += 0.1
            
        self.consciousness_state['attention_radius'] = base_radius + pulse
        
    def apply_attention_mask(self, frame):
        """Apply consciousness-based attention mask to frame"""
        h, w = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        focus_x = int(self.consciousness_state['focus_x'] * w)
        focus_y = int(self.consciousness_state['focus_y'] * h)
        radius = int(self.consciousness_state['attention_radius'] * min(h, w))
        
        # Create attention mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Multiple attention layers
        cv2.circle(mask, (focus_x, focus_y), radius, 255, -1)
        cv2.circle(mask, (focus_x, focus_y), int(radius * 1.5), 200, int(radius * 0.3))
        cv2.circle(mask, (focus_x, focus_y), int(radius * 2.5), 100, int(radius * 0.5))
        
        # Blur the mask
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        # Apply mask to frame
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        attended_frame = cv2.addWeighted(frame, 0.3, mask_3channel, 0.7, 0)
        
        # Draw consciousness indicators
        self.draw_consciousness_overlay(attended_frame, focus_x, focus_y, radius)
        
        return attended_frame
        
    def draw_consciousness_overlay(self, frame, focus_x, focus_y, radius):
        """Draw consciousness state visualization"""
        h, w = frame.shape[:2]
        
        # Draw attention center
        cv2.circle(frame, (focus_x, focus_y), 5, (0, 255, 255), -1)
        
        # Draw attention boundary
        cv2.circle(frame, (focus_x, focus_y), radius, (0, 255, 0), 2)
        
        # Draw peak motion indicator if active
        if self.peak_motion_region['strength'] > 0.01:
            peak_x = int(self.peak_motion_region['x'] * w)
            peak_y = int(self.peak_motion_region['y'] * h)
            peak_size = int(20 * self.peak_motion_region['strength'] / 0.03)
            cv2.circle(frame, (peak_x, peak_y), peak_size, (0, 0, 255), 2)
            cv2.putText(frame, "PEAK", (peak_x - 20, peak_y - peak_size - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Status text
        curiosity_text = f"Curiosity: {self.consciousness_state['curiosity']:.2f}"
        cv2.putText(frame, curiosity_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Motion statistics
        if self.motion_history:
            current_motion = self.motion_history[-1]
            avg_motion = np.mean(self.motion_history)
            peak_motion = max(self.motion_history)
            pa_ratio = peak_motion / max(self.ambient_motion, 0.001)
            
            # Motion text with P/A ratio
            motion_text = f"Motion: {current_motion:.3f} | P/A: {pa_ratio:.1f}"
            
            # Color based on P/A ratio
            if pa_ratio > 3.0:
                color = (0, 0, 255)  # Red for high P/A
            elif pa_ratio > 1.5:
                color = (0, 255, 255)  # Yellow for medium P/A
            else:
                color = (255, 255, 255)  # White for low P/A
                
            cv2.putText(frame, motion_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # P/A ratio bar
            bar_width = int(200 * min(1.0, pa_ratio / 5.0))
            cv2.rectangle(frame, (10, 110), (210, 125), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 110), (10 + bar_width, 125), color, -1)
            cv2.putText(frame, "P/A Ratio", (220, 122),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Sustained motion indicator
            if self.sustained_motion_frames > self.motion_persistence_threshold:
                cv2.putText(frame, "SUSTAINED MOTION", (10, 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # AI DNA influence
        if self.ai_dna_patterns:
            dna_text = f"DNA Patterns: {len(self.ai_dna_patterns)}"
            cv2.putText(frame, dna_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
    def analyze_frame(self, frame, prev_frame=None):
        """Enhanced frame analysis with motion detection"""
        stats = {}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness
        stats['brightness'] = np.mean(gray)
        
        # Enhanced motion detection
        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Frame difference
            diff = cv2.absdiff(gray, prev_gray)
            
            # Threshold to reduce noise
            _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Dilate to connect motion regions
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            
            # Count motion pixels
            motion_pixels = np.sum(thresh > 0)
            stats['motion_pixels'] = motion_pixels
            
            # Calculate motion magnitude
            stats['motion'] = np.mean(diff) / 255.0
            
            # Find motion center with weighted average
            if motion_pixels > 100:
                # Find contours of motion regions
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest contour (main motion)
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    
                    if M['m00'] > 0:
                        cx = M['m10'] / M['m00'] / frame.shape[1]
                        cy = M['m01'] / M['m00'] / frame.shape[0]
                        stats['motion_center'] = (cx, cy)
                    else:
                        stats['motion_center'] = (0.5, 0.5)
                else:
                    stats['motion_center'] = (0.5, 0.5)
            else:
                stats['motion_center'] = (0.5, 0.5)
            
            # Visual motion indicator
            motion_viz = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            motion_viz[:, :, 0] = 0  # Remove blue
            motion_viz[:, :, 1] = 0  # Remove green
            # Red shows motion
            
            # Overlay motion indicator
            motion_small = cv2.resize(motion_viz, (160, 120))
            frame[10:130, frame.shape[1]-170:frame.shape[1]-10] = motion_small
            cv2.rectangle(frame, (frame.shape[1]-170, 10), (frame.shape[1]-10, 130), (255, 0, 0), 2)
            cv2.putText(frame, "Motion", (frame.shape[1]-160, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        else:
            stats['motion'] = 0
            stats['motion_pixels'] = 0
            stats['motion_center'] = (0.5, 0.5)
            
        return stats

def main():
    print("🧠 Consciousness-Guided Visual Attention - Enhanced")
    print("=" * 40)
    print("Peak motion tracking with P/A ratio response")
    print("High P/A ratio = strong focus snap")
    print("Low P/A ratio = gentle wandering")
    print("Press 'q' to quit, 's' to save snapshot")
    print("=" * 40)
    
    # Initialize consciousness
    consciousness = ConsciousnessAttention()
    
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
            
        # Analyze frame
        stats = consciousness.analyze_frame(frame, prev_frame)
        
        # Update consciousness state
        consciousness.update_consciousness(stats)
        
        # Apply attention mask
        attended_frame = consciousness.apply_attention_mask(frame)
        
        # Display
        cv2.imshow('Consciousness Vision - Enhanced', attended_frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"consciousness_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, attended_frame)
            print(f"Saved: {filename}")
            
        prev_frame = frame.copy()
        frame_count += 1
        
        # Print stats occasionally
        if frame_count % 90 == 0:
            pa_ratio = max(consciousness.motion_history) / consciousness.ambient_motion if consciousness.motion_history else 1.0
            print(f"Frame {frame_count} | Focus: ({consciousness.consciousness_state['focus_x']:.2f}, "
                  f"{consciousness.consciousness_state['focus_y']:.2f}) | "
                  f"P/A Ratio: {pa_ratio:.1f} | "
                  f"Peak: ({consciousness.peak_motion_region['x']:.2f}, {consciousness.peak_motion_region['y']:.2f})")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nConsciousness vision ended")

if __name__ == "__main__":
    main()