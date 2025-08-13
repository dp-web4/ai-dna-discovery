#!/usr/bin/env python3
"""
Interactive Camera Trust Test - Legion
With live display window and obscuring test capability
"""
import cv2
import time
import numpy as np
from camera_trust import camera_trust_score

def run_interactive_trust_test():
    """Run interactive camera trust test with display window"""
    
    print("=" * 60)
    print("Interactive Camera Trust Test - Legion")
    print("=" * 60)
    print("\nInstructions:")
    print("- Live camera feed will appear in a window")
    print("- Try covering the camera with your hand")
    print("- Watch trust score change in real-time")
    print("- Press 'q' or ESC to quit")
    print("- Press 's' to save a snapshot")
    print("=" * 60)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    prev_gray = None
    frame_count = 0
    snapshot_count = 0
    
    # For tracking min/max trust
    min_trust = 1.0
    max_trust = 0.0
    
    print("\nStarting interactive test...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame")
                continue
            
            # Calculate trust score
            score, metrics = camera_trust_score(frame, prev_gray=prev_gray, resize_w=320)
            
            # Update min/max
            min_trust = min(min_trust, score)
            max_trust = max(max_trust, score)
            
            # Prepare for next iteration
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = gray
            
            frame_count += 1
            
            # Create display frame with overlay
            display = frame.copy()
            
            # Add trust score bar
            bar_height = 30
            bar_width = int(display.shape[1] * 0.8)
            bar_x = int(display.shape[1] * 0.1)
            bar_y = 30
            
            # Background
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Trust bar (color based on score)
            if score > 0.7:
                color = (0, 255, 0)  # Green - good
            elif score > 0.4:
                color = (0, 165, 255)  # Orange - medium
            else:
                color = (0, 0, 255)  # Red - poor
            
            trust_width = int(bar_width * score)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + trust_width, bar_y + bar_height),
                         color, -1)
            
            # Add text overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Trust score
            cv2.putText(display, f"Trust: {score:.3f}", (bar_x, bar_y - 5),
                       font, 0.6, (255, 255, 255), 2)
            
            # Min/Max tracker
            cv2.putText(display, f"Min: {min_trust:.3f} | Max: {max_trust:.3f}", 
                       (bar_x + bar_width - 200, bar_y - 5),
                       font, 0.5, (200, 200, 200), 1)
            
            # Component metrics
            y_offset = 80
            metrics_text = [
                f"Sharpness: {metrics['tenengrad']:.1f}",
                f"Edges: {metrics['edge_density']:.3f}",
                f"Contrast: {metrics['rms_contrast']:.3f}",
                f"Saturation: {metrics['sat_mean']:.3f}",
                f"Noise: {metrics['spatial_noise']:.3f}",
                f"Exposure: L={metrics['low_clip']:.2f} H={metrics['high_clip']:.2f}"
            ]
            
            for i, text in enumerate(metrics_text):
                cv2.putText(display, text, (20, y_offset + i*25),
                           font, 0.5, (255, 255, 255), 1)
            
            # Camera state indicator
            state_y = display.shape[0] - 50
            if score < 0.1:
                state_text = "CAMERA OBSCURED!"
                state_color = (0, 0, 255)
            elif score < 0.3:
                state_text = "Poor visibility"
                state_color = (0, 165, 255)
            elif score < 0.5:
                state_text = "Limited visibility"
                state_color = (0, 255, 255)
            elif score < 0.7:
                state_text = "Good visibility"
                state_color = (0, 255, 0)
            else:
                state_text = "Excellent visibility"
                state_color = (0, 255, 0)
            
            cv2.putText(display, state_text, (20, state_y),
                       font, 0.8, state_color, 2)
            
            # Frame counter
            cv2.putText(display, f"Frame: {frame_count}", 
                       (display.shape[1] - 120, display.shape[0] - 20),
                       font, 0.5, (200, 200, 200), 1)
            
            # Instructions
            cv2.putText(display, "Press 'q' to quit | 's' to snapshot | Cover camera to test", 
                       (20, display.shape[0] - 20),
                       font, 0.4, (200, 200, 200), 1)
            
            # Show the frame
            cv2.imshow("Camera Trust Test - Legion", display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # Save snapshot
                snapshot_count += 1
                filename = f"trust_snapshot_{snapshot_count}_{score:.3f}.jpg"
                cv2.imwrite(filename, display)
                print(f"Saved snapshot: {filename} (trust={score:.3f})")
            
            # Print periodic updates to console
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: trust={score:.3f} | state: {state_text}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Final summary
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Trust range observed: {min_trust:.3f} to {max_trust:.3f}")
    print(f"Snapshots saved: {snapshot_count}")
    
    if min_trust < 0.1:
        print("✓ Camera obscuring detected successfully!")
    
    print("=" * 60)

if __name__ == "__main__":
    run_interactive_trust_test()