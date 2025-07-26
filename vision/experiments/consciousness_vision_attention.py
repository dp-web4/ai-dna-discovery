#!/usr/bin/env python3
"""
Consciousness-Guided Visual Attention
Combines AI DNA patterns with vision to create dynamic attention regions
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
            'focus_x': 0.5,  # 0-1 normalized
            'focus_y': 0.5,
            'attention_radius': 0.3,
            'curiosity': 0.5,
            'pattern_memory': []
        }
        
        # Load AI DNA patterns if available
        self.ai_dna_patterns = self.load_ai_dna_patterns()
        
    def load_ai_dna_patterns(self):
        """Load patterns from AI DNA experiments"""
        patterns = []
        dna_path = Path("../../ai_dna_results")
        
        if dna_path.exists():
            # Load a few DNA cycles for pattern inspiration
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
        # Simulate consciousness drift based on visual entropy
        motion = frame_stats.get('motion', 0)
        brightness_change = frame_stats.get('brightness_change', 0)
        
        # Curiosity increases with motion
        self.consciousness_state['curiosity'] = min(1.0, 
            self.consciousness_state['curiosity'] * 0.95 + motion * 0.05)
        
        # Focus drifts with AI DNA pattern influence
        if self.ai_dna_patterns and random.random() < 0.1:
            # Occasionally jump attention based on AI DNA
            pattern = random.choice(self.ai_dna_patterns)
            # Convert pattern to attention coordinates
            pattern_hash = hash(str(pattern))
            self.consciousness_state['focus_x'] = (pattern_hash % 100) / 100.0
            self.consciousness_state['focus_y'] = ((pattern_hash // 100) % 100) / 100.0
        else:
            # Smooth drift
            drift_x = (random.random() - 0.5) * 0.02
            drift_y = (random.random() - 0.5) * 0.02
            
            self.consciousness_state['focus_x'] = np.clip(
                self.consciousness_state['focus_x'] + drift_x, 0.1, 0.9)
            self.consciousness_state['focus_y'] = np.clip(
                self.consciousness_state['focus_y'] + drift_y, 0.1, 0.9)
        
        # Attention radius pulses with consciousness
        pulse = np.sin(time.time() * 2) * 0.05
        self.consciousness_state['attention_radius'] = 0.3 + pulse + (self.consciousness_state['curiosity'] * 0.2)
        
    def apply_attention_mask(self, frame):
        """Apply consciousness-based attention mask to frame"""
        h, w = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        focus_x = int(self.consciousness_state['focus_x'] * w)
        focus_y = int(self.consciousness_state['focus_y'] * h)
        radius = int(self.consciousness_state['attention_radius'] * min(h, w))
        
        # Create attention mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Multiple attention layers inspired by consciousness levels
        # Primary focus
        cv2.circle(mask, (focus_x, focus_y), radius, 255, -1)
        
        # Secondary awareness ring
        cv2.circle(mask, (focus_x, focus_y), int(radius * 1.5), 200, int(radius * 0.3))
        
        # Peripheral vision
        cv2.circle(mask, (focus_x, focus_y), int(radius * 2.5), 100, int(radius * 0.5))
        
        # Blur the mask for smooth transitions
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
        
        # Show AI DNA influence
        if self.ai_dna_patterns:
            dna_text = f"DNA Patterns: {len(self.ai_dna_patterns)}"
            cv2.putText(frame, dna_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
    def analyze_frame(self, frame, prev_frame=None):
        """Analyze frame for consciousness-relevant features"""
        stats = {}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness
        stats['brightness'] = np.mean(gray)
        
        # Motion detection if previous frame available
        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)
            stats['motion'] = np.mean(diff) / 255.0
            stats['brightness_change'] = abs(stats['brightness'] - np.mean(prev_gray)) / 255.0
        else:
            stats['motion'] = 0
            stats['brightness_change'] = 0
            
        return stats

def main():
    print("🧠 Consciousness-Guided Visual Attention")
    print("=" * 40)
    print("Watch as AI consciousness guides visual focus!")
    print("Press 'q' to quit, 's' to save snapshot")
    print("=" * 40)
    
    # Initialize consciousness
    consciousness = ConsciousnessAttention()
    
    # Open camera
    cap = cv2.VideoCapture(0)  # Will use V4L2 backend
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
        
    # Set camera properties for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
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
        cv2.imshow('Consciousness Vision', attended_frame)
        
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
                  f"Curiosity: {consciousness.consciousness_state['curiosity']:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nConsciousness vision ended")

if __name__ == "__main__":
    main()