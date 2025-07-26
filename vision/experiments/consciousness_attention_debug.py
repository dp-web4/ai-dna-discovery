#!/usr/bin/env python3
"""
Consciousness-Guided Visual Attention with Motion Debug
Enhanced version with visual motion feedback
"""

import cv2
import numpy as np
import json
import time
import random
from pathlib import Path

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
        self.motion_threshold = 5  # Lowered threshold
        self.motion_history = []
        self.max_history = 30
        
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
        
    def update_consciousness(self, frame_stats):
        """Update consciousness state based on visual input"""
        motion = frame_stats.get('motion', 0)
        motion_pixels = frame_stats.get('motion_pixels', 0)
        
        # Store motion history
        self.motion_history.append(motion)
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)
        
        # Use motion value directly (0.016 ambient, 0.045 with motion)
        # Scale so 0.02 = low motion, 0.04+ = high motion
        motion_factor = max(0, (motion - 0.015) * 20)  # Subtract ambient, amplify
        motion_factor = min(1.0, motion_factor)
        
        # Curiosity decays naturally, spikes with motion
        self.consciousness_state['curiosity'] = min(1.0, 
            self.consciousness_state['curiosity'] * 0.85 + motion_factor * 0.15)
        
        # Clamp minimum curiosity so it can recover
        if self.consciousness_state['curiosity'] < 0.2:
            self.consciousness_state['curiosity'] = 0.2
        
        # Focus can jump toward motion
        # Lower threshold and use motion value instead of pixels
        if motion > 0.025 and random.random() < 0.5:  # 50% chance when motion detected
            # Jump toward motion center more aggressively
            motion_center = frame_stats.get('motion_center', (0.5, 0.5))
            self.consciousness_state['focus_x'] = (
                self.consciousness_state['focus_x'] * 0.5 + motion_center[0] * 0.5)
            self.consciousness_state['focus_y'] = (
                self.consciousness_state['focus_y'] * 0.5 + motion_center[1] * 0.5)
        elif self.ai_dna_patterns and random.random() < 0.05:
            # AI DNA influence (less frequent)
            pattern = random.choice(self.ai_dna_patterns)
            pattern_hash = hash(str(pattern))
            self.consciousness_state['focus_x'] = (pattern_hash % 100) / 100.0
            self.consciousness_state['focus_y'] = ((pattern_hash // 100) % 100) / 100.0
        else:
            # Smooth drift (larger when curious)
            drift_scale = 0.01 + (self.consciousness_state['curiosity'] * 0.02)
            drift_x = (random.random() - 0.5) * drift_scale
            drift_y = (random.random() - 0.5) * drift_scale
            
            self.consciousness_state['focus_x'] = np.clip(
                self.consciousness_state['focus_x'] + drift_x, 0.1, 0.9)
            self.consciousness_state['focus_y'] = np.clip(
                self.consciousness_state['focus_y'] + drift_y, 0.1, 0.9)
        
        # Attention radius pulses with consciousness
        pulse = np.sin(time.time() * 2) * 0.05
        base_radius = 0.25 + (self.consciousness_state['curiosity'] * 0.3)
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
        # Draw attention center
        cv2.circle(frame, (focus_x, focus_y), 5, (0, 255, 255), -1)
        
        # Draw attention boundary
        cv2.circle(frame, (focus_x, focus_y), radius, (0, 255, 0), 2)
        
        # Draw consciousness state text
        curiosity_text = f"Curiosity: {self.consciousness_state['curiosity']:.2f}"
        cv2.putText(frame, curiosity_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show motion level with better visualization
        if self.motion_history:
            current_motion = self.motion_history[-1] if self.motion_history else 0
            avg_motion = np.mean(self.motion_history[-10:])  # Last 10 frames
            motion_text = f"Motion: {current_motion:.3f} (avg: {avg_motion:.3f})"
            
            # Color based on motion level
            if current_motion > 0.04:
                color = (0, 0, 255)  # Red for high motion
            elif current_motion > 0.025:
                color = (0, 255, 255)  # Yellow for medium motion
            else:
                color = (255, 255, 255)  # White for low/no motion
                
            cv2.putText(frame, motion_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw motion indicator bar
            bar_width = int(200 * min(1.0, current_motion / 0.05))
            cv2.rectangle(frame, (10, 110), (210, 125), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 110), (10 + bar_width, 125), color, -1)
            cv2.putText(frame, "Motion Level", (220, 122),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show AI DNA influence
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
            
            # Count motion pixels
            motion_pixels = np.sum(thresh > 0)
            stats['motion_pixels'] = motion_pixels
            
            # Calculate motion magnitude
            stats['motion'] = np.mean(diff) / 255.0
            
            # Find motion center (for attention guidance)
            if motion_pixels > 100:
                moments = cv2.moments(thresh)
                if moments['m00'] > 0:
                    cx = moments['m10'] / moments['m00'] / frame.shape[1]
                    cy = moments['m01'] / moments['m00'] / frame.shape[0]
                    stats['motion_center'] = (cx, cy)
                else:
                    stats['motion_center'] = (0.5, 0.5)
            else:
                stats['motion_center'] = (0.5, 0.5)
            
            stats['brightness_change'] = abs(stats['brightness'] - np.mean(prev_gray)) / 255.0
            
            # Visual motion indicator
            motion_viz = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            motion_viz[:, :, 0] = 0  # Remove blue channel
            motion_viz[:, :, 1] = 0  # Remove green channel
            # Red shows motion
            
            # Overlay motion indicator on frame (top-right corner, small)
            motion_small = cv2.resize(motion_viz, (160, 120))
            frame[10:130, frame.shape[1]-170:frame.shape[1]-10] = motion_small
            
            # Draw motion indicator border
            cv2.rectangle(frame, (frame.shape[1]-170, 10), (frame.shape[1]-10, 130), (255, 0, 0), 2)
            cv2.putText(frame, "Motion", (frame.shape[1]-160, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        else:
            stats['motion'] = 0
            stats['motion_pixels'] = 0
            stats['motion_center'] = (0.5, 0.5)
            stats['brightness_change'] = 0
            
        return stats

def main():
    print("🧠 Consciousness-Guided Visual Attention (Debug Version)")
    print("=" * 40)
    print("Enhanced motion detection with visual feedback")
    print("Red overlay in top-right shows motion")
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
            
        # Analyze frame (this also adds motion visualization)
        stats = consciousness.analyze_frame(frame, prev_frame)
        
        # Update consciousness state
        consciousness.update_consciousness(stats)
        
        # Apply attention mask
        attended_frame = consciousness.apply_attention_mask(frame)
        
        # Display
        cv2.imshow('Consciousness Vision (Debug)', attended_frame)
        
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
            print(f"Frame {frame_count} | Focus: ({consciousness.consciousness_state['focus_x']:.2f}, "
                  f"{consciousness.consciousness_state['focus_y']:.2f}) | "
                  f"Curiosity: {consciousness.consciousness_state['curiosity']:.2f} | "
                  f"Motion: {stats.get('motion_pixels', 0)} pixels")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nConsciousness vision ended")

if __name__ == "__main__":
    main()