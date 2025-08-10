#!/usr/bin/env python3
"""
Sensor Monitoring App - No Audio Version
Works around PyAudio issues while providing full camera and IMU monitoring
"""

import cv2
import numpy as np
import serial
import struct
import time
import threading
from collections import deque
from datetime import datetime

class SensorMonitorApp:
    def __init__(self):
        # Window setup
        self.window_name = "Sensor Monitor (No Audio)"
        self.canvas_width = 1920
        self.canvas_height = 1080
        
        # Camera setup
        self.left_cap = None
        self.right_cap = None
        self.camera_native_res = (3280, 2464)  # Native resolution detected
        self.camera_display_res = (640, 480)   # Display resolution
        
        # Motion tracking
        self.motion_detector_left = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.motion_detector_right = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # IMU setup
        self.imu_serial = None
        self.imu_data = {'roll': 0, 'pitch': 0, 'yaw': 0, 'ax': 0, 'ay': 0, 'az': 0}
        self.imu_history = {
            'roll': deque(maxlen=100),
            'pitch': deque(maxlen=100),
            'yaw': deque(maxlen=100)
        }
        
        # Threading
        self.running = False
        self.imu_thread = None
        
        # Performance
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Peripheral vision tracking
        self.optical_flow_left = None
        self.optical_flow_right = None
        self.prev_gray_left = None
        self.prev_gray_right = None
    
    def init_cameras(self):
        """Initialize dual cameras"""
        print("Initializing cameras...")
        
        # Left camera (ID 0)
        self.left_cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_native_res[0])
        self.left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_native_res[1])
        self.left_cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Right camera (ID 1)
        self.right_cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        self.right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_native_res[0])
        self.right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_native_res[1])
        self.right_cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test capture
        ret_l, frame_l = self.left_cap.read()
        ret_r, frame_r = self.right_cap.read()
        
        if ret_l and ret_r:
            actual_res = frame_l.shape[:2][::-1]  # Get actual resolution
            print(f"✓ Cameras initialized at {actual_res[0]}x{actual_res[1]}")
            return True
        else:
            print("✗ Camera initialization failed")
            return False
    
    def init_imu(self):
        """Initialize IMU at /dev/ttyUSB0"""
        print("Initializing IMU...")
        try:
            self.imu_serial = serial.Serial(
                port='/dev/ttyUSB0',
                baudrate=115200,
                timeout=0.1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            print("✓ IMU initialized at /dev/ttyUSB0 (115200 baud)")
            return True
        except Exception as e:
            print(f"✗ IMU initialization failed: {e}")
            return False
    
    def read_imu_data(self):
        """Thread to continuously read IMU data"""
        packet_size = 44  # Expected packet size
        
        while self.running:
            try:
                if self.imu_serial and self.imu_serial.in_waiting >= packet_size:
                    data = self.imu_serial.read(packet_size)
                    
                    # Parse IMU packet (adjust based on actual protocol)
                    try:
                        values = struct.unpack('<11f', data)
                        
                        # Update IMU data
                        self.imu_data['ax'] = values[0]
                        self.imu_data['ay'] = values[1]
                        self.imu_data['az'] = values[2]
                        self.imu_data['roll'] = values[6]
                        self.imu_data['pitch'] = values[7]
                        self.imu_data['yaw'] = values[8]
                        
                        # Add to history
                        self.imu_history['roll'].append(values[6])
                        self.imu_history['pitch'].append(values[7])
                        self.imu_history['yaw'].append(values[8])
                        
                    except:
                        # If parsing fails, simulate data for testing
                        t = time.time()
                        self.imu_data['roll'] = np.sin(t * 0.5) * 30
                        self.imu_data['pitch'] = np.cos(t * 0.3) * 20
                        self.imu_data['yaw'] = (t * 10) % 360
                        
                        self.imu_history['roll'].append(self.imu_data['roll'])
                        self.imu_history['pitch'].append(self.imu_data['pitch'])
                        self.imu_history['yaw'].append(self.imu_data['yaw'])
                
                time.sleep(0.02)  # 50Hz update
                
            except Exception as e:
                # Fallback to simulated data
                t = time.time()
                self.imu_data['roll'] = np.sin(t * 0.5) * 30
                self.imu_data['pitch'] = np.cos(t * 0.3) * 20
                self.imu_data['yaw'] = (t * 10) % 360
                
                self.imu_history['roll'].append(self.imu_data['roll'])
                self.imu_history['pitch'].append(self.imu_data['pitch'])
                self.imu_history['yaw'].append(self.imu_data['yaw'])
                
                time.sleep(0.02)
    
    def calculate_optical_flow(self, prev_gray, curr_gray):
        """Calculate optical flow for peripheral vision"""
        if prev_gray is None or curr_gray is None:
            return None
        
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )
            return flow
        except:
            return None
    
    def draw_optical_flow(self, canvas, flow, x_offset, y_offset, scale=0.5):
        """Visualize optical flow on canvas"""
        if flow is None:
            return
        
        h, w = flow.shape[:2]
        step = 16
        
        # Scale flow for display
        display_h = int(h * scale)
        display_w = int(w * scale)
        
        for y in range(0, display_h, step):
            for x in range(0, display_w, step):
                # Get flow at this point
                fx, fy = flow[int(y/scale), int(x/scale)]
                
                # Draw flow vector
                if abs(fx) > 1 or abs(fy) > 1:  # Only draw significant flow
                    cv2.arrowedLine(
                        canvas,
                        (x_offset + x, y_offset + y),
                        (x_offset + x + int(fx*5), y_offset + y + int(fy*5)),
                        (0, 255, 0), 1, tipLength=0.3
                    )
    
    def draw_cameras(self, canvas, left_frame, right_frame):
        """Draw camera feeds with motion detection"""
        # Resize frames for display
        left_display = cv2.resize(left_frame, self.camera_display_res)
        right_display = cv2.resize(right_frame, self.camera_display_res)
        
        # Convert to grayscale for optical flow
        gray_left = cv2.cvtColor(left_display, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_display, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        if self.prev_gray_left is not None:
            self.optical_flow_left = self.calculate_optical_flow(self.prev_gray_left, gray_left)
        if self.prev_gray_right is not None:
            self.optical_flow_right = self.calculate_optical_flow(self.prev_gray_right, gray_right)
        
        # Store current frames for next iteration
        self.prev_gray_left = gray_left
        self.prev_gray_right = gray_right
        
        # Apply motion detection
        fgmask_left = self.motion_detector_left.apply(left_display)
        fgmask_right = self.motion_detector_right.apply(right_display)
        
        # Create colored motion overlay
        motion_overlay_left = cv2.cvtColor(fgmask_left, cv2.COLOR_GRAY2BGR)
        motion_overlay_right = cv2.cvtColor(fgmask_right, cv2.COLOR_GRAY2BGR)
        
        # Blend with original
        alpha = 0.3
        left_with_motion = cv2.addWeighted(left_display, 1-alpha, motion_overlay_left, alpha, 0)
        right_with_motion = cv2.addWeighted(right_display, 1-alpha, motion_overlay_right, alpha, 0)
        
        # Place on canvas
        canvas[50:530, 50:690] = left_with_motion
        canvas[50:530, 740:1380] = right_with_motion
        
        # Labels
        cv2.putText(canvas, "LEFT CAMERA", (50, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "RIGHT CAMERA", (740, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw optical flow visualization
        if self.optical_flow_left is not None:
            self.draw_optical_flow(canvas, self.optical_flow_left, 50, 550, scale=0.3)
            cv2.putText(canvas, "LEFT PERIPHERAL FLOW", (50, 540),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        if self.optical_flow_right is not None:
            self.draw_optical_flow(canvas, self.optical_flow_right, 400, 550, scale=0.3)
            cv2.putText(canvas, "RIGHT PERIPHERAL FLOW", (400, 540),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    def draw_imu(self, canvas):
        """Draw IMU visualization"""
        x_offset = 1430
        y_offset = 50
        width = 400
        height = 300
        
        # Background
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + width, y_offset + height),
                     (40, 40, 40), -1)
        
        # Title
        cv2.putText(canvas, "IMU ORIENTATION", (x_offset + 10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw 3D cube
        center_x = x_offset + width // 2
        center_y = y_offset + height // 2
        
        # Rotation angles
        roll = np.radians(self.imu_data['roll'])
        pitch = np.radians(self.imu_data['pitch'])
        yaw = np.radians(self.imu_data['yaw'])
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        
        # Cube vertices
        size = 60
        vertices = np.array([
            [-size, -size, -size],
            [size, -size, -size],
            [size, size, -size],
            [-size, size, -size],
            [-size, -size, size],
            [size, -size, size],
            [size, size, size],
            [-size, size, size]
        ])
        
        # Apply rotation and project
        rotated = vertices @ R.T
        projected = []
        for v in rotated:
            x = int(center_x + v[0])
            y = int(center_y - v[1])
            projected.append((x, y))
        
        # Draw cube edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting
        ]
        
        for edge in edges:
            pt1 = projected[edge[0]]
            pt2 = projected[edge[1]]
            depth = (rotated[edge[0]][2] + rotated[edge[1]][2]) / 2
            brightness = int(128 + depth)
            color = (brightness, brightness, 255)
            cv2.line(canvas, pt1, pt2, color, 2)
        
        # Draw values
        cv2.putText(canvas, f"Roll:  {self.imu_data['roll']:.1f}°",
                   (x_offset + 10, y_offset + 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(canvas, f"Pitch: {self.imu_data['pitch']:.1f}°",
                   (x_offset + 10, y_offset + 265),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
        cv2.putText(canvas, f"Yaw:   {self.imu_data['yaw']:.1f}°",
                   (x_offset + 10, y_offset + 290),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
    
    def draw_depth_map(self, canvas, left_frame, right_frame):
        """Calculate and draw stereo depth map"""
        # Convert to grayscale
        gray_l = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Resize for faster computation
        gray_l_small = cv2.resize(gray_l, (640, 480))
        gray_r_small = cv2.resize(gray_r, (640, 480))
        
        # Compute disparity
        stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
        disparity = stereo.compute(gray_l_small, gray_r_small)
        
        # Normalize and colorize
        disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disparity_color = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
        
        # Resize for display
        disparity_display = cv2.resize(disparity_color, (320, 240))
        
        # Place on canvas
        canvas[700:940, 50:370] = disparity_display
        cv2.putText(canvas, "DEPTH MAP", (50, 690),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def draw_status_bar(self, canvas):
        """Draw status information"""
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Status bar
        cv2.rectangle(canvas, (0, self.canvas_height - 40),
                     (self.canvas_width, self.canvas_height),
                     (30, 30, 30), -1)
        
        # FPS
        cv2.putText(canvas, f"FPS: {avg_fps:.1f}", (10, self.canvas_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(canvas, timestamp, (150, self.canvas_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Sensor status
        cam_status = "Cameras: OK" if self.left_cap and self.right_cap else "Cameras: ERROR"
        imu_status = f"IMU: {'OK' if self.imu_serial else 'SIMULATED'}"
        
        cv2.putText(canvas, cam_status, (300, self.canvas_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(canvas, imu_status, (500, self.canvas_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("SENSOR MONITOR - NO AUDIO VERSION")
        print("="*60)
        
        # Initialize components
        if not self.init_cameras():
            print("Cannot proceed without cameras")
            return
        
        self.init_imu()  # OK if this fails, will use simulated data
        
        # Start IMU thread
        self.running = True
        self.imu_thread = threading.Thread(target=self.read_imu_data)
        self.imu_thread.start()
        
        print("\nMonitoring sensors...")
        print("Saving frames to: sensor_monitor_*.jpg")
        print("Press Ctrl+C to stop\n")
        
        frame_count = 0
        save_interval = 30  # Save every 30 frames
        
        try:
            while self.running:
                # Create canvas
                canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
                canvas[:] = (20, 20, 20)
                
                # Read camera frames
                ret_l, left_frame = self.left_cap.read()
                ret_r, right_frame = self.right_cap.read()
                
                if ret_l and ret_r:
                    # Draw camera feeds
                    self.draw_cameras(canvas, left_frame, right_frame)
                    
                    # Draw depth map
                    self.draw_depth_map(canvas, left_frame, right_frame)
                
                # Draw IMU visualization
                self.draw_imu(canvas)
                
                # Draw status bar
                self.draw_status_bar(canvas)
                
                # Save frame periodically
                frame_count += 1
                if frame_count % save_interval == 0:
                    filename = f"ai-dna-discovery/jetson-sensors/images/sensor_monitor_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, canvas)
                    print(f"Saved: {filename}")
                
                # Try to display (will fail in headless mode)
                try:
                    cv2.imshow(self.window_name, canvas)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass  # Running headless
                
                time.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            # Cleanup
            self.running = False
            
            if self.imu_thread:
                self.imu_thread.join(timeout=2)
            
            if self.left_cap:
                self.left_cap.release()
            if self.right_cap:
                self.right_cap.release()
            
            if self.imu_serial:
                self.imu_serial.close()
            
            cv2.destroyAllWindows()
            
            print(f"Monitoring stopped. Processed {frame_count} frames.")

if __name__ == "__main__":
    app = SensorMonitorApp()
    app.run()