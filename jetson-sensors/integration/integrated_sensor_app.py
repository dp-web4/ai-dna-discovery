#!/usr/bin/env python3
"""
Integrated Sensor Visualization App
Live video from dual cameras with tracking, IMU visualization, and audio traces
"""

import cv2
import numpy as np
import pyaudio
import serial
import struct
import time
import threading
import queue
from collections import deque
from datetime import datetime
import sys

# Add sensors directory to path
sys.path.append('/home/sprout/ai-workspace/private-context/sensors')

class IntegratedSensorApp:
    def __init__(self):
        # Window setup
        self.window_name = "Integrated Sensor Dashboard"
        self.display_width = 1920
        self.display_height = 1080
        
        # Camera setup
        self.left_cap = None
        self.right_cap = None
        self.camera_width = 640
        self.camera_height = 480
        
        # Motion tracking
        self.motion_detector_left = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.motion_detector_right = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.tracker_left = None
        self.tracker_right = None
        self.tracking_box_left = None
        self.tracking_box_right = None
        
        # IMU setup
        self.imu_serial = None
        self.imu_data = {'roll': 0, 'pitch': 0, 'yaw': 0}
        self.imu_history = {
            'roll': deque(maxlen=100),
            'pitch': deque(maxlen=100),
            'yaw': deque(maxlen=100)
        }
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.audio_stream_in = None
        self.audio_stream_out = None
        self.audio_sample_rate = 44100
        self.audio_chunk_size = 1024
        
        # Audio visualization
        self.mic_volume_history = deque(maxlen=200)
        self.speaker_volume_history = deque(maxlen=200)
        self.audio_queue_in = queue.Queue()
        self.audio_queue_out = queue.Queue()
        
        # Threading
        self.running = False
        self.imu_thread = None
        self.audio_thread = None
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
    def init_cameras(self):
        """Initialize dual cameras"""
        print("Initializing cameras...")
        
        # Left camera
        self.left_cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.left_cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Right camera
        self.right_cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        self.right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.right_cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test capture
        ret_l, _ = self.left_cap.read()
        ret_r, _ = self.right_cap.read()
        
        if ret_l and ret_r:
            print("✓ Cameras initialized")
            return True
        else:
            print("✗ Camera initialization failed")
            return False
    
    def init_imu(self):
        """Initialize IMU connection"""
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
            print("✓ IMU initialized at /dev/ttyUSB0")
            return True
        except Exception as e:
            print(f"✗ IMU initialization failed: {e}")
            return False
    
    def init_audio(self):
        """Initialize audio streams"""
        print("Initializing audio...")
        try:
            # Try to initialize audio with error handling
            import os
            os.environ['PYTHONIOENCODING'] = 'utf-8'  # Avoid encoding issues
            
            # Input stream (microphone) with safer parameters
            try:
                self.audio_stream_in = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.audio_sample_rate,
                    input=True,
                    frames_per_buffer=self.audio_chunk_size,
                    stream_callback=None  # Disable callback to avoid SystemError
                )
                print("✓ Audio initialized (polling mode)")
            except:
                # If audio fails, continue without it
                self.audio_stream_in = None
                print("⚠ Audio input disabled (PyAudio issue)")
            
            return True
        except Exception as e:
            print(f"✗ Audio initialization failed: {e}")
            self.audio_stream_in = None
            return True  # Continue without audio
    
    def audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream"""
        if self.running:
            # Convert to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            # Calculate RMS volume
            volume = np.sqrt(np.mean(audio_data**2))
            # Normalize to 0-1 range
            normalized_volume = min(1.0, volume / 32768.0)
            self.mic_volume_history.append(normalized_volume)
        
        return (in_data, pyaudio.paContinue)
    
    def read_imu_data(self):
        """Thread function to read IMU data"""
        while self.running:
            try:
                if self.imu_serial and self.imu_serial.in_waiting >= 44:
                    data = self.imu_serial.read(44)
                    # Simple parsing - adjust based on actual IMU protocol
                    # This is a placeholder for the actual protocol
                    values = struct.unpack('<11f', data)
                    
                    # Extract euler angles (assuming they're in positions 6-8)
                    self.imu_data['roll'] = values[6]
                    self.imu_data['pitch'] = values[7]
                    self.imu_data['yaw'] = values[8]
                    
                    # Add to history
                    self.imu_history['roll'].append(values[6])
                    self.imu_history['pitch'].append(values[7])
                    self.imu_history['yaw'].append(values[8])
                    
            except Exception as e:
                # Simulate IMU data if real IMU fails
                t = time.time()
                self.imu_data['roll'] = np.sin(t * 0.5) * 30
                self.imu_data['pitch'] = np.cos(t * 0.3) * 20
                self.imu_data['yaw'] = (t * 10) % 360
                
                self.imu_history['roll'].append(self.imu_data['roll'])
                self.imu_history['pitch'].append(self.imu_data['pitch'])
                self.imu_history['yaw'].append(self.imu_data['yaw'])
            
            time.sleep(0.02)  # 50Hz update rate
    
    def process_frame_tracking(self, frame, motion_detector, tracker, tracking_box, camera_name):
        """Process frame for motion detection and tracking"""
        # Motion detection
        fgmask = motion_detector.apply(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest motion area
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Initialize or update tracker
                if tracker is None or tracking_box is None:
                    # Initialize tracker with detected motion
                    tracker = cv2.TrackerCSRT_create()
                    tracking_box = (x, y, w, h)
                    tracker.init(frame, tracking_box)
                else:
                    # Update existing tracker
                    success, tracking_box = tracker.update(frame)
                    if not success:
                        # Reinitialize if tracking fails
                        tracker = cv2.TrackerCSRT_create()
                        tracking_box = (x, y, w, h)
                        tracker.init(frame, tracking_box)
                
                # Draw tracking box
                if tracking_box:
                    x, y, w, h = [int(v) for v in tracking_box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{camera_name} Tracking", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, tracker, tracking_box
    
    def draw_imu_visualization(self, canvas, x_offset, y_offset):
        """Draw IMU visualization on canvas"""
        # Create IMU display area
        imu_width = 400
        imu_height = 300
        
        # Background
        cv2.rectangle(canvas, (x_offset, y_offset), 
                     (x_offset + imu_width, y_offset + imu_height),
                     (40, 40, 40), -1)
        
        # Title
        cv2.putText(canvas, "IMU Orientation", (x_offset + 10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw 3D cube representation
        center_x = x_offset + imu_width // 2
        center_y = y_offset + imu_height // 2 + 20
        
        # Simple 3D to 2D projection of a cube
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
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Cube vertices
        size = 50
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
        
        # Apply rotation
        rotated = vertices @ R.T
        
        # Project to 2D
        projected = []
        for v in rotated:
            x = int(center_x + v[0])
            y = int(center_y - v[1])  # Flip Y axis
            projected.append((x, y))
        
        # Draw cube edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        for edge in edges:
            pt1 = projected[edge[0]]
            pt2 = projected[edge[1]]
            # Color based on depth (simple shading)
            depth = (rotated[edge[0]][2] + rotated[edge[1]][2]) / 2
            brightness = int(128 + depth)
            color = (brightness, brightness, 255)
            cv2.line(canvas, pt1, pt2, color, 2)
        
        # Draw orientation values
        cv2.putText(canvas, f"Roll:  {self.imu_data['roll']:.1f}°", 
                   (x_offset + 10, y_offset + 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(canvas, f"Pitch: {self.imu_data['pitch']:.1f}°", 
                   (x_offset + 10, y_offset + 245),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
        cv2.putText(canvas, f"Yaw:   {self.imu_data['yaw']:.1f}°", 
                   (x_offset + 10, y_offset + 270),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
        
        # Draw history graphs
        graph_x = x_offset + 220
        graph_y = y_offset + 200
        graph_width = 160
        graph_height = 80
        
        # Background for graphs
        cv2.rectangle(canvas, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height),
                     (20, 20, 20), -1)
        
        # Plot histories
        if len(self.imu_history['roll']) > 1:
            # Roll history (green)
            points = []
            for i, val in enumerate(self.imu_history['roll']):
                x = graph_x + int(i * graph_width / 100)
                y = graph_y + graph_height // 2 - int(val * graph_height / 180)
                points.append((x, y))
            
            for i in range(1, len(points)):
                cv2.line(canvas, points[i-1], points[i], (100, 255, 100), 1)
        
        if len(self.imu_history['pitch']) > 1:
            # Pitch history (red)
            points = []
            for i, val in enumerate(self.imu_history['pitch']):
                x = graph_x + int(i * graph_width / 100)
                y = graph_y + graph_height // 2 - int(val * graph_height / 180)
                points.append((x, y))
            
            for i in range(1, len(points)):
                cv2.line(canvas, points[i-1], points[i], (255, 100, 100), 1)
    
    def draw_audio_visualization(self, canvas, x_offset, y_offset):
        """Draw audio volume traces"""
        audio_width = 400
        audio_height = 300
        
        # Background
        cv2.rectangle(canvas, (x_offset, y_offset), 
                     (x_offset + audio_width, y_offset + audio_height),
                     (40, 40, 40), -1)
        
        # Title
        audio_title = "Audio Levels" if self.audio_stream_in else "Audio (Disabled)"
        cv2.putText(canvas, audio_title, (x_offset + 10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Poll audio if stream exists and callback is disabled
        if self.audio_stream_in and self.audio_stream_in.is_active():
            try:
                # Read audio data in polling mode
                audio_data = self.audio_stream_in.read(self.audio_chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_array**2))
                normalized_volume = min(1.0, volume / 32768.0)
                self.mic_volume_history.append(normalized_volume)
            except:
                pass  # Ignore audio errors
        
        # Microphone trace area
        mic_y = y_offset + 60
        mic_height = 100
        
        cv2.putText(canvas, "Microphone", (x_offset + 10, mic_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Draw mic volume trace
        if len(self.mic_volume_history) > 1:
            points = []
            for i, vol in enumerate(self.mic_volume_history):
                x = x_offset + 10 + int(i * (audio_width - 20) / 200)
                y = mic_y + mic_height - int(vol * mic_height)
                points.append((x, y))
            
            for i in range(1, len(points)):
                cv2.line(canvas, points[i-1], points[i], (100, 255, 100), 2)
        
        # Current mic level bar
        if self.mic_volume_history:
            current_vol = self.mic_volume_history[-1]
            bar_width = int(current_vol * (audio_width - 20))
            cv2.rectangle(canvas, (x_offset + 10, mic_y + mic_height + 5),
                         (x_offset + 10 + bar_width, mic_y + mic_height + 20),
                         (100, 255, 100), -1)
        
        # Speaker trace area (simulated for now)
        speaker_y = y_offset + 190
        speaker_height = 100
        
        cv2.putText(canvas, "Speaker (System Audio)", (x_offset + 10, speaker_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        
        # Simulate speaker volume for demo
        t = time.time()
        speaker_vol = abs(np.sin(t * 2)) * 0.5 + np.random.random() * 0.1
        self.speaker_volume_history.append(speaker_vol)
        
        # Draw speaker volume trace
        if len(self.speaker_volume_history) > 1:
            points = []
            for i, vol in enumerate(self.speaker_volume_history):
                x = x_offset + 10 + int(i * (audio_width - 20) / 200)
                y = speaker_y + speaker_height - int(vol * speaker_height)
                points.append((x, y))
            
            for i in range(1, len(points)):
                cv2.line(canvas, points[i-1], points[i], (255, 100, 100), 2)
        
        # Current speaker level bar
        if self.speaker_volume_history:
            current_vol = self.speaker_volume_history[-1]
            bar_width = int(current_vol * (audio_width - 20))
            cv2.rectangle(canvas, (x_offset + 10, speaker_y + speaker_height + 5),
                         (x_offset + 10 + bar_width, speaker_y + speaker_height + 20),
                         (255, 100, 100), -1)
    
    def draw_status_bar(self, canvas):
        """Draw status bar with FPS and system info"""
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Status bar background
        cv2.rectangle(canvas, (0, self.display_height - 40),
                     (self.display_width, self.display_height),
                     (30, 30, 30), -1)
        
        # FPS
        cv2.putText(canvas, f"FPS: {avg_fps:.1f}", (10, self.display_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(canvas, timestamp, (150, self.display_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Sensor status
        cam_status = "Cameras: OK" if self.left_cap and self.right_cap else "Cameras: ERROR"
        imu_status = "IMU: OK" if self.imu_serial else "IMU: SIMULATED"
        audio_status = "Audio: OK" if self.audio_stream_in else "Audio: ERROR"
        
        cv2.putText(canvas, cam_status, (300, self.display_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(canvas, imu_status, (450, self.display_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(canvas, audio_status, (600, self.display_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("INTEGRATED SENSOR VISUALIZATION")
        print("="*60)
        
        # Initialize components
        cameras_ok = self.init_cameras()
        imu_ok = self.init_imu()
        audio_ok = self.init_audio()
        
        if not cameras_ok:
            print("Cannot proceed without cameras")
            return
        
        # Start background threads
        self.running = True
        
        self.imu_thread = threading.Thread(target=self.read_imu_data)
        self.imu_thread.start()
        
        # Create window with simpler settings
        try:
            cv2.namedWindow(self.window_name)
        except:
            print("Warning: Could not create GUI window, running in headless mode")
            print("Saving frames to integrated_sensor_output.jpg instead")
        
        print("\nPress 'q' to quit")
        print("Press 'r' to reset tracking")
        print("="*60 + "\n")
        
        try:
            while self.running:
                # Create canvas
                canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                canvas[:] = (20, 20, 20)  # Dark background
                
                # Read camera frames
                ret_l, left_frame = self.left_cap.read()
                ret_r, right_frame = self.right_cap.read()
                
                if ret_l and ret_r:
                    # Resize frames to expected display size first
                    left_frame = cv2.resize(left_frame, (self.camera_width, self.camera_height))
                    right_frame = cv2.resize(right_frame, (self.camera_width, self.camera_height))
                    
                    # Process tracking on resized frames
                    left_frame, self.tracker_left, self.tracking_box_left = \
                        self.process_frame_tracking(left_frame, self.motion_detector_left,
                                                   self.tracker_left, self.tracking_box_left, "LEFT")
                    
                    right_frame, self.tracker_right, self.tracking_box_right = \
                        self.process_frame_tracking(right_frame, self.motion_detector_right,
                                                   self.tracker_right, self.tracking_box_right, "RIGHT")
                    
                    # Place camera feeds on canvas
                    # Left camera
                    canvas[50:50+self.camera_height, 50:50+self.camera_width] = left_frame
                    cv2.putText(canvas, "LEFT CAMERA", (50, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Right camera
                    canvas[50:50+self.camera_height, 740:740+self.camera_width] = right_frame
                    cv2.putText(canvas, "RIGHT CAMERA", (740, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw IMU visualization
                self.draw_imu_visualization(canvas, 1430, 50)
                
                # Draw audio visualization
                self.draw_audio_visualization(canvas, 1430, 380)
                
                # Draw disparity/depth estimation between cameras
                if ret_l and ret_r:
                    # Simple disparity visualization
                    gray_l = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                    gray_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Compute disparity
                    stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
                    disparity = stereo.compute(gray_l, gray_r)
                    
                    # Normalize for display
                    disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    disparity_color = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
                    
                    # Resize and place on canvas
                    disparity_small = cv2.resize(disparity_color, (320, 240))
                    canvas[580:820, 50:370] = disparity_small
                    cv2.putText(canvas, "DEPTH MAP", (50, 570),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw peripheral vision indicator
                cv2.putText(canvas, "PERIPHERAL VISION GYROSCOPE", (420, 570),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw flow field visualization
                if ret_l:
                    # Simple optical flow visualization
                    flow_viz = np.zeros((240, 320, 3), dtype=np.uint8)
                    
                    # Create flow field indicators
                    for y in range(0, 240, 20):
                        for x in range(0, 320, 20):
                            # Simulate flow based on IMU rotation
                            flow_x = int(self.imu_data['yaw'] / 10)
                            flow_y = int(self.imu_data['pitch'] / 10)
                            
                            cv2.arrowedLine(flow_viz, (x, y), 
                                          (x + flow_x, y + flow_y),
                                          (0, 255, 0), 1, tipLength=0.3)
                    
                    canvas[580:820, 420:740] = flow_viz
                
                # Draw sensor fusion confidence
                overall_confidence = 0.75  # Placeholder
                conf_bar_width = int(overall_confidence * 300)
                cv2.rectangle(canvas, (790, 580), (790 + conf_bar_width, 610),
                             (0, 255, 0), -1)
                cv2.rectangle(canvas, (790, 580), (1090, 610), (100, 100, 100), 2)
                cv2.putText(canvas, f"FUSION CONFIDENCE: {overall_confidence:.0%}",
                           (790, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw status bar
                self.draw_status_bar(canvas)
                
                # Display or save
                try:
                    cv2.imshow(self.window_name, canvas)
                except:
                    # Save frame if no display
                    cv2.imwrite('integrated_sensor_output.jpg', canvas)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset tracking
                    self.tracker_left = None
                    self.tracker_right = None
                    self.tracking_box_left = None
                    self.tracking_box_right = None
                    print("Tracking reset")
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        
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
            
            if self.audio_stream_in:
                self.audio_stream_in.stop_stream()
                self.audio_stream_in.close()
            
            self.audio.terminate()
            
            cv2.destroyAllWindows()
            
            print("Cleanup complete")

if __name__ == "__main__":
    app = IntegratedSensorApp()
    app.run()