#!/usr/bin/env python3
"""
Stereo Vision Test - Display both cameras side by side
Tests dual IMX219 camera setup on Jetson Orin Nano
"""

import cv2
import numpy as np
import threading
import queue
import time

class StereoVision:
    def __init__(self):
        self.frame_queue_0 = queue.Queue(maxsize=2)
        self.frame_queue_1 = queue.Queue(maxsize=2)
        self.running = True
        
    def camera_pipeline(self, sensor_id, width=640, height=480, fps=30):
        """GStreamer pipeline for each camera"""
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"framerate={fps}/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1"
        )
    
    def capture_thread(self, sensor_id, frame_queue):
        """Thread to capture from one camera"""
        pipeline = self.camera_pipeline(sensor_id)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print(f"Failed to open camera {sensor_id}")
            return
            
        print(f"Camera {sensor_id} started")
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Drop old frames if queue is full
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                frame_queue.put(frame)
                
        cap.release()
        print(f"Camera {sensor_id} stopped")
    
    def run(self):
        """Main stereo vision display"""
        # Start capture threads
        thread0 = threading.Thread(target=self.capture_thread, args=(0, self.frame_queue_0))
        thread1 = threading.Thread(target=self.capture_thread, args=(1, self.frame_queue_1))
        
        thread0.start()
        thread1.start()
        
        print("\nStereo Vision Test")
        print("==================")
        print("Press 'q' to quit")
        print("Press 's' to save stereo pair")
        print("Press 'd' to compute disparity (simple)")
        
        frame_count = 0
        fps_timer = time.time()
        show_disparity = False
        
        while True:
            try:
                # Get frames from both cameras
                frame0 = self.frame_queue_0.get(timeout=1)
                frame1 = self.frame_queue_1.get(timeout=1)
                
                # Add labels
                cv2.putText(frame0, "Camera 0 (Left)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame1, "Camera 1 (Right)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Combine side by side
                stereo_frame = np.hstack([frame0, frame1])
                
                # Optional: Simple disparity calculation
                if show_disparity:
                    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    
                    # Simple block matching
                    stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
                    disparity = stereo.compute(gray0, gray1)
                    disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    disparity_color = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
                    
                    # Show disparity below stereo view
                    combined = np.vstack([stereo_frame, 
                                        np.hstack([disparity_color, disparity_color])])
                    cv2.imshow('Stereo Vision + Disparity', combined)
                else:
                    cv2.imshow('Stereo Vision', stereo_frame)
                
                # FPS calculation
                frame_count += 1
                if time.time() - fps_timer > 1.0:
                    fps = frame_count / (time.time() - fps_timer)
                    print(f"FPS: {fps:.1f}")
                    frame_count = 0
                    fps_timer = time.time()
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"stereo_left_{timestamp}.jpg", frame0)
                    cv2.imwrite(f"stereo_right_{timestamp}.jpg", frame1)
                    print(f"Saved stereo pair: {timestamp}")
                elif key == ord('d'):
                    show_disparity = not show_disparity
                    print(f"Disparity: {'ON' if show_disparity else 'OFF'}")
                    
            except queue.Empty:
                print("Frame timeout - check cameras")
                continue
            except Exception as e:
                print(f"Error: {e}")
                break
        
        # Cleanup
        self.running = False
        thread0.join()
        thread1.join()
        cv2.destroyAllWindows()

def main():
    print("🎥🎥 Dual Camera Stereo Vision Test")
    print("===================================")
    
    stereo = StereoVision()
    
    try:
        stereo.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    print("Test complete!")

if __name__ == "__main__":
    main()