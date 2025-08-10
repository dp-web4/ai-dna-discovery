#!/usr/bin/env python3
"""
Vision Sensor with Peripheral Gyroscope Functionality
Dual CSI camera system with motion detection and attention mechanisms
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from collections import deque
import threading

@dataclass
class VisionData:
    """Vision sensor data with peripheral gyroscope"""
    timestamp: float
    left_frame: Optional[np.ndarray]
    right_frame: Optional[np.ndarray]
    optical_flow: Dict  # Peripheral motion vectors
    motion_regions: List[Tuple[int, int, int, int]]  # Detected motion areas
    attention_point: Optional[Tuple[int, int]]  # Current focus point
    peripheral_stability: float  # 0.0 to 1.0 (gyroscope confidence)
    context_state: str  # 'stable', 'turning', 'moving', 'unstable'
    confidence: float  # Overall vision confidence

class PeripheralGyroscope:
    """Peripheral vision as gyroscopic reference"""
    
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define peripheral zones (outer edges of vision)
        self.peripheral_zones = {
            'left': (0, 0, int(frame_width * 0.2), frame_height),
            'right': (int(frame_width * 0.8), 0, frame_width, frame_height),
            'top': (0, 0, frame_width, int(frame_height * 0.2)),
            'bottom': (0, int(frame_height * 0.8), frame_width, frame_height)
        }
        
        # Flow history for stability calculation
        self.flow_history = deque(maxlen=10)
        
    def calculate_optical_flow(self, prev_gray, curr_gray) -> Dict:
        """Calculate optical flow in peripheral zones"""
        flow_data = {}
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )
        
        # Analyze flow in each peripheral zone
        for zone_name, (x1, y1, x2, y2) in self.peripheral_zones.items():
            zone_flow = flow[y1:y2, x1:x2]
            
            # Calculate average flow magnitude and direction
            mag, ang = cv2.cartToPolar(zone_flow[:,:,0], zone_flow[:,:,1])
            
            flow_data[zone_name] = {
                'magnitude': np.mean(mag),
                'direction': np.mean(ang),
                'std_dev': np.std(mag),  # Variability indicates instability
                'max_magnitude': np.max(mag)
            }
        
        # Calculate overall flow pattern
        flow_data['overall'] = self.interpret_flow_pattern(flow_data)
        
        return flow_data
    
    def interpret_flow_pattern(self, flow_data: Dict) -> Dict:
        """Interpret optical flow pattern as motion type"""
        left_mag = flow_data['left']['magnitude']
        right_mag = flow_data['right']['magnitude']
        top_mag = flow_data['top']['magnitude']
        bottom_mag = flow_data['bottom']['magnitude']
        
        # Horizontal flow analysis (turning detection)
        horizontal_diff = left_mag - right_mag
        vertical_diff = top_mag - bottom_mag
        
        # Determine motion type
        if abs(horizontal_diff) > 2.0:
            if horizontal_diff > 0:
                motion_type = 'turning_right'  # Left periphery moving more
            else:
                motion_type = 'turning_left'   # Right periphery moving more
        elif np.mean([left_mag, right_mag, top_mag, bottom_mag]) > 3.0:
            motion_type = 'moving_forward'
        elif np.mean([left_mag, right_mag, top_mag, bottom_mag]) < 0.5:
            motion_type = 'stationary'
        else:
            motion_type = 'complex_motion'
        
        return {
            'type': motion_type,
            'horizontal_flow': horizontal_diff,
            'vertical_flow': vertical_diff,
            'average_magnitude': np.mean([left_mag, right_mag, top_mag, bottom_mag])
        }
    
    def calculate_stability(self, flow_data: Dict) -> float:
        """Calculate peripheral stability (gyroscope confidence)"""
        # Add current flow to history
        if not flow_data or 'overall' not in flow_data:
            return 1.0  # Default to stable if no flow data
        
        self.flow_history.append(flow_data['overall']['average_magnitude'])
        
        if len(self.flow_history) < 2:
            return 1.0
        
        # Calculate stability based on flow consistency
        flow_variance = np.var(self.flow_history)
        
        # Check for edge detection (sudden drops in peripheral flow)
        edge_detected = False
        for zone_name in ['left', 'right', 'top', 'bottom']:
            if flow_data[zone_name]['std_dev'] > 5.0:  # High variability
                edge_detected = True
                break
        
        # Calculate stability score
        if edge_detected:
            stability = 0.3  # Low stability near edges/drops
        elif flow_variance > 10.0:
            stability = 0.5  # Medium stability with high variance
        elif flow_variance > 5.0:
            stability = 0.7  # Good stability with moderate variance
        else:
            stability = 0.9  # Excellent stability with low variance
        
        return stability

class VisionSensor:
    """Dual camera vision system with attention and peripheral gyroscope"""
    
    def __init__(self, left_camera_id=0, right_camera_id=1):
        self.left_camera_id = left_camera_id
        self.right_camera_id = right_camera_id
        
        self.left_cap = None
        self.right_cap = None
        self.is_connected = False
        
        # Frame buffers
        self.prev_left_gray = None
        self.prev_right_gray = None
        
        # Peripheral gyroscope
        self.peripheral_gyro = PeripheralGyroscope()
        
        # Motion detection
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True
        )
        
        # Attention system
        self.attention_point = None
        self.attention_history = deque(maxlen=30)
        
        # Confidence tracking
        self.confidence_history = deque(maxlen=10)
        
        # Context state
        self.context_state = 'stable'
        self.context_weights = {
            'stable': {'peripheral': 0.8, 'central': 0.2, 'motion': 0.3},
            'turning': {'peripheral': 0.9, 'central': 0.1, 'motion': 0.5},
            'moving': {'peripheral': 0.7, 'central': 0.3, 'motion': 0.7},
            'unstable': {'peripheral': 0.3, 'central': 0.7, 'motion': 0.9}
        }
    
    def connect(self) -> bool:
        """Connect to dual cameras"""
        try:
            # Open left camera
            self.left_cap = cv2.VideoCapture(self.left_camera_id, cv2.CAP_V4L2)
            self.left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.left_cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Open right camera
            self.right_cap = cv2.VideoCapture(self.right_camera_id, cv2.CAP_V4L2)
            self.right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.right_cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret_l, _ = self.left_cap.read()
            ret_r, _ = self.right_cap.read()
            
            if ret_l and ret_r:
                self.is_connected = True
                print("Connected to dual CSI cameras")
                return True
            else:
                print("Failed to read from cameras")
                return False
                
        except Exception as e:
            print(f"Failed to connect to cameras: {e}")
            return False
    
    def detect_motion(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect motion regions in frame"""
        # Apply background subtraction
        fgmask = self.motion_detector.apply(frame)
        
        # Find contours
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and get bounding boxes
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x, y, w, h))
        
        return motion_regions
    
    def update_attention(self, motion_regions: List, peripheral_flow: Dict):
        """Update attention point based on motion and peripheral flow"""
        if not motion_regions:
            # No motion, gradually return to center
            if self.attention_point:
                center = (320, 240)
                self.attention_point = (
                    int(self.attention_point[0] * 0.9 + center[0] * 0.1),
                    int(self.attention_point[1] * 0.9 + center[1] * 0.1)
                )
            return
        
        # Find most salient region (largest motion)
        largest_area = 0
        salient_point = None
        
        for x, y, w, h in motion_regions:
            area = w * h
            if area > largest_area:
                largest_area = area
                salient_point = (x + w//2, y + h//2)
        
        # Check if peripheral flow suggests redirection
        flow_pattern = peripheral_flow.get('overall', {})
        if flow_pattern.get('type') == 'turning_left':
            # Bias attention to the right
            if salient_point:
                salient_point = (min(salient_point[0] + 50, 640), salient_point[1])
        elif flow_pattern.get('type') == 'turning_right':
            # Bias attention to the left
            if salient_point:
                salient_point = (max(salient_point[0] - 50, 0), salient_point[1])
        
        # Smooth attention movement
        if self.attention_point and salient_point:
            self.attention_point = (
                int(self.attention_point[0] * 0.7 + salient_point[0] * 0.3),
                int(self.attention_point[1] * 0.7 + salient_point[1] * 0.3)
            )
        elif salient_point:
            self.attention_point = salient_point
        
        # Add to history
        if self.attention_point:
            self.attention_history.append(self.attention_point)
    
    def update_context_state(self, peripheral_stability: float, motion_count: int):
        """Update context state based on sensor inputs"""
        if peripheral_stability < 0.4:
            self.context_state = 'unstable'
        elif motion_count > 5:
            self.context_state = 'moving'
        elif peripheral_stability > 0.8 and motion_count < 2:
            self.context_state = 'stable'
        else:
            self.context_state = 'turning'
    
    def calculate_confidence(self, peripheral_stability: float, 
                            frame_quality: float, motion_detected: bool) -> float:
        """Calculate overall vision confidence"""
        weights = self.context_weights[self.context_state]
        
        # Component confidences
        peripheral_conf = peripheral_stability * weights['peripheral']
        central_conf = frame_quality * weights['central']
        motion_conf = (0.8 if motion_detected else 0.3) * weights['motion']
        
        # Weighted average
        total_weight = sum(weights.values())
        confidence = (peripheral_conf + central_conf + motion_conf) / total_weight
        
        # Temporal smoothing
        self.confidence_history.append(confidence)
        return np.mean(self.confidence_history)
    
    def assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess frame quality (blur, exposure, etc)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Check blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 100)  # Normalize
        
        # Check exposure
        mean_brightness = np.mean(gray)
        exposure_score = 1.0 - abs(mean_brightness - 128) / 128
        
        return (blur_score + exposure_score) / 2
    
    def read(self) -> Optional[VisionData]:
        """Read and process vision data"""
        if not self.is_connected:
            return None
        
        # Capture frames
        ret_l, left_frame = self.left_cap.read()
        ret_r, right_frame = self.right_cap.read()
        
        if not ret_l or not ret_r:
            return None
        
        # Convert to grayscale for processing
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate peripheral optical flow
        optical_flow = {}
        if self.prev_left_gray is not None:
            optical_flow = self.peripheral_gyro.calculate_optical_flow(
                self.prev_left_gray, left_gray
            )
        
        # Calculate peripheral stability
        peripheral_stability = self.peripheral_gyro.calculate_stability(optical_flow)
        
        # Detect motion in left frame (primary)
        motion_regions = self.detect_motion(left_frame)
        
        # Update attention
        self.update_attention(motion_regions, optical_flow)
        
        # Update context state
        self.update_context_state(peripheral_stability, len(motion_regions))
        
        # Assess frame quality
        frame_quality = self.assess_frame_quality(left_frame)
        
        # Calculate overall confidence
        confidence = self.calculate_confidence(
            peripheral_stability, frame_quality, len(motion_regions) > 0
        )
        
        # Store current frames for next iteration
        self.prev_left_gray = left_gray
        self.prev_right_gray = right_gray
        
        # Create vision data object
        return VisionData(
            timestamp=time.time(),
            left_frame=left_frame,
            right_frame=right_frame,
            optical_flow=optical_flow,
            motion_regions=motion_regions,
            attention_point=self.attention_point,
            peripheral_stability=peripheral_stability,
            context_state=self.context_state,
            confidence=confidence
        )
    
    def get_sensor_fusion_data(self) -> Dict:
        """Get data formatted for sensor fusion system"""
        data = self.read()
        if not data:
            return None
        
        return {
            'type': 'vision',
            'timestamp': data.timestamp,
            'data': {
                'peripheral_gyroscope': {
                    'stability': data.peripheral_stability,
                    'flow': data.optical_flow
                },
                'attention': {
                    'point': data.attention_point,
                    'context': data.context_state
                },
                'motion': {
                    'regions': data.motion_regions,
                    'count': len(data.motion_regions)
                }
            },
            'confidence': data.confidence,
            'metadata': {
                'cameras': 'Dual IMX219 CSI',
                'resolution': '640x480',
                'fps': 30
            }
        }
    
    def close(self):
        """Close camera connections"""
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap.release()
        self.is_connected = False
        print("Vision sensors closed")

if __name__ == "__main__":
    # Test vision sensor
    vision = VisionSensor()
    
    if vision.connect():
        print("\nReading vision data...")
        print("-" * 60)
        
        try:
            for i in range(300):  # 10 seconds at 30 FPS
                data = vision.read()
                if data:
                    print(f"Time: {data.timestamp:.2f}")
                    print(f"Context: {data.context_state}")
                    print(f"Peripheral Stability: {data.peripheral_stability:.2%}")
                    print(f"Motion Regions: {len(data.motion_regions)}")
                    if data.attention_point:
                        print(f"Attention: {data.attention_point}")
                    print(f"Confidence: {data.confidence:.2%}")
                    
                    # Show flow pattern if available
                    if data.optical_flow and 'overall' in data.optical_flow:
                        flow = data.optical_flow['overall']
                        print(f"Flow Pattern: {flow.get('type', 'unknown')}")
                    
                    print("-" * 60)
                    time.sleep(0.033)  # ~30 FPS
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            vision.close()