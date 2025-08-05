#!/usr/bin/env python3
"""
Camera Confidence Implementation
Evaluates stereo camera confidence for binocular vision system
"""

import cv2
import numpy as np
import time
from confidence_framework import CameraConfidence
from typing import Dict, Tuple, Optional

class StereoCameraConfidence:
    """Confidence evaluation for stereo camera pair"""
    
    def __init__(self, left_camera=0, right_camera=1):
        self.left_conf = BinocularCameraConfidence("left", left_camera)
        self.right_conf = BinocularCameraConfidence("right", right_camera)
        self.stereo_metrics = {}
        
    def audit_stereo_pair(self) -> Dict[str, float]:
        """Audit both cameras and their stereo relationship"""
        print("🔍 Auditing Stereo Camera Pair")
        print("=" * 40)
        
        # Individual camera audits
        left_audit = self.left_conf.audit()
        right_audit = self.right_conf.audit()
        
        # Stereo-specific tests
        stereo_audit = self._audit_stereo_alignment()
        
        # Combine results
        combined_audit = {
            'left_camera': left_audit,
            'right_camera': right_audit,
            'stereo_alignment': stereo_audit
        }
        
        return combined_audit
    
    def _audit_stereo_alignment(self) -> Dict[str, float]:
        """Test stereo camera alignment and calibration"""
        print("\n📐 Testing stereo alignment...")
        
        # Capture frames from both cameras
        left_frame = self.left_conf.capture_frame()
        right_frame = self.right_conf.capture_frame()
        
        if left_frame is None or right_frame is None:
            return {'alignment': 0.0, 'disparity_quality': 0.0}
        
        # Convert to grayscale for stereo matching
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Check if images are similar (same scene)
        similarity = self._compute_similarity(left_gray, right_gray)
        
        # Try to compute disparity
        disparity_quality = self._test_disparity_computation(left_gray, right_gray)
        
        # Check temporal sync (capture time difference)
        time_sync = self._test_temporal_sync()
        
        results = {
            'similarity': similarity,
            'disparity_quality': disparity_quality,
            'temporal_sync': time_sync,
            'alignment': (similarity + disparity_quality + time_sync) / 3
        }
        
        print(f"  Stereo similarity: {similarity:.0%}")
        print(f"  Disparity quality: {disparity_quality:.0%}")
        print(f"  Temporal sync: {time_sync:.0%}")
        
        return results
    
    def _compute_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute similarity between left and right images"""
        if left.shape != right.shape:
            return 0.0
        
        # Compute normalized cross-correlation
        left_norm = (left - left.mean()) / left.std()
        right_norm = (right - right.mean()) / right.std()
        
        correlation = np.corrcoef(left_norm.flatten(), right_norm.flatten())[0, 1]
        
        # Convert to 0-1 confidence score
        return max(0, min(1, (correlation + 1) / 2))
    
    def _test_disparity_computation(self, left: np.ndarray, right: np.ndarray) -> float:
        """Test quality of stereo disparity computation"""
        try:
            # Create stereo matcher
            stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
            
            # Compute disparity
            disparity = stereo.compute(left, right)
            
            # Check disparity quality
            valid_pixels = np.sum(disparity > 0)
            total_pixels = disparity.size
            
            if total_pixels > 0:
                valid_ratio = valid_pixels / total_pixels
                return min(valid_ratio * 2, 1.0)  # Scale up to reward good matches
            
        except Exception as e:
            print(f"    Disparity computation failed: {e}")
            
        return 0.0
    
    def _test_temporal_sync(self) -> float:
        """Test temporal synchronization between cameras"""
        # Capture multiple frames and check timing
        timestamps = []
        
        for _ in range(5):
            start = time.time()
            left_frame = self.left_conf.capture_frame()
            mid = time.time()
            right_frame = self.right_conf.capture_frame()
            end = time.time()
            
            if left_frame is not None and right_frame is not None:
                timestamps.append((mid - start, end - mid))
        
        if timestamps:
            # Check consistency of capture times
            left_times, right_times = zip(*timestamps)
            time_consistency = 1.0 - min(np.std(left_times) + np.std(right_times), 1.0)
            return time_consistency
        
        return 0.5

class BinocularCameraConfidence(CameraConfidence):
    """Individual camera confidence for binocular vision"""
    
    def __init__(self, eye_name: str, camera_id: int):
        super().__init__(f"{eye_name}_eye")
        self.eye_name = eye_name
        self.camera_id = camera_id
        self.cap = None
        
    def audit(self) -> Dict[str, float]:
        """Audit individual camera capabilities"""
        print(f"\n👁️  Auditing {self.eye_name} camera (ID: {self.camera_id})")
        
        audit_results = {}
        
        # Test camera connection
        connection_quality = self._test_connection()
        audit_results['connection'] = connection_quality
        
        if connection_quality < 0.5:
            print(f"    ❌ Camera connection failed")
            return audit_results
        
        # Test image quality
        image_quality = self._test_image_quality()
        audit_results['image_quality'] = image_quality
        
        # Test frame rate
        fps_stability = self._test_fps_stability()
        audit_results['fps_stability'] = fps_stability
        
        # Test auto-focus/exposure
        auto_features = self._test_auto_features()
        audit_results.update(auto_features)
        
        print(f"    Connection: {connection_quality:.0%}")
        print(f"    Image Quality: {image_quality:.0%}")
        print(f"    FPS Stability: {fps_stability:.0%}")
        
        return audit_results
    
    def _test_connection(self) -> float:
        """Test camera connection and basic functionality"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                return 0.0
            
            # Try to capture a frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return 0.3
            
            # Check frame properties
            height, width = frame.shape[:2]
            if height < 100 or width < 100:
                return 0.5
            
            return 1.0
            
        except Exception as e:
            print(f"    Connection error: {e}")
            return 0.0
    
    def _test_image_quality(self) -> float:
        """Test image quality metrics"""
        frame = self.capture_frame()
        if frame is None:
            return 0.0
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check brightness
        brightness = np.mean(gray)
        brightness_score = 1.0 - min(abs(brightness - 128) / 128, 1.0)
        
        # Check contrast (standard deviation)
        contrast = np.std(gray)
        contrast_score = min(contrast / 50, 1.0)
        
        # Check for motion blur (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_quality = min(blur_score / 100, 1.0)
        
        # Combine scores
        quality = (brightness_score + contrast_score + blur_quality) / 3
        return quality
    
    def _test_fps_stability(self) -> float:
        """Test frame rate stability"""
        if not self.cap:
            return 0.0
        
        frame_times = []
        start_time = time.time()
        
        # Capture frames for 2 seconds
        while time.time() - start_time < 2.0:
            frame_start = time.time()
            ret, frame = self.cap.read()
            if ret:
                frame_times.append(time.time() - frame_start)
        
        if len(frame_times) < 10:
            return 0.0
        
        # Calculate FPS stability
        intervals = np.diff(frame_times)
        if len(intervals) > 0:
            stability = 1.0 - min(np.std(intervals) / np.mean(intervals), 1.0)
            fps = len(frame_times) / 2.0
            
            # Penalize if FPS is too low
            fps_score = min(fps / 25.0, 1.0)  # Expect ~25-30 FPS
            
            return (stability + fps_score) / 2
        
        return 0.0
    
    def _test_auto_features(self) -> Dict[str, float]:
        """Test auto-exposure, auto-focus, etc."""
        results = {}
        
        # Capture multiple frames to test adaptation
        frames = []
        for _ in range(10):
            frame = self.capture_frame()
            if frame is not None:
                frames.append(frame)
            time.sleep(0.1)
        
        if frames:
            # Test exposure adaptation
            brightnesses = [np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
            exposure_stability = 1.0 - min(np.std(brightnesses) / np.mean(brightnesses), 1.0)
            results['auto_exposure'] = exposure_stability
            
            # Test focus consistency (edge sharpness)
            sharpness = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            focus_stability = 1.0 - min(np.std(sharpness) / np.mean(sharpness), 1.0)
            results['auto_focus'] = focus_stability
        else:
            results['auto_exposure'] = 0.0
            results['auto_focus'] = 0.0
        
        return results
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if not self.cap:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def evaluate_context(self, context: Dict) -> float:
        """Evaluate camera relevance in current context"""
        relevance = 0.1  # Base relevance
        
        # Vision tasks
        if context.get('vision_active', False):
            relevance += 0.7
        
        # Stereo vision specifically
        if context.get('stereo_vision', False):
            relevance += 0.2
        
        # Lighting conditions
        lighting = context.get('lighting', 'normal')
        if lighting == 'good':
            relevance += 0.2
        elif lighting == 'poor':
            relevance -= 0.3
        
        # Motion detection
        if context.get('motion_detection', False):
            relevance += 0.3
            
        return max(0, min(relevance, 1.0))
    
    def __del__(self):
        """Clean up camera resources"""
        if self.cap:
            self.cap.release()

# Example usage
if __name__ == "__main__":
    print("🎥 Camera Confidence Audit")
    
    # Test stereo cameras
    stereo = StereoCameraConfidence(0, 1)
    results = stereo.audit_stereo_pair()
    
    print("\n📊 Stereo Camera Audit Results:")
    print(f"Left camera overall: {np.mean(list(results['left_camera'].values())):.0%}")
    print(f"Right camera overall: {np.mean(list(results['right_camera'].values())):.0%}")
    print(f"Stereo alignment: {results['stereo_alignment']['alignment']:.0%}")
    
    # Clean up
    del stereo