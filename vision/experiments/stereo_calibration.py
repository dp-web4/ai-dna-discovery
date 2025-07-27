#!/usr/bin/env python3
"""
Stereo Calibration Helper
Handles misalignment between fixed cameras
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime

class StereoCalibrator:
    """Calibrate stereo camera pair for misalignment"""
    
    def __init__(self, checkerboard_size=(9, 6), square_size_mm=25):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size_mm
        
        # Calibration data
        self.calibration_data = {
            'baseline_mm': 76.2,  # 3 inches = 76.2mm
            'left_camera': {},
            'right_camera': {},
            'stereo': {},
            'timestamp': None
        }
        
        # Calibration images
        self.left_images = []
        self.right_images = []
        
    def capture_calibration_pair(self, left_frame, right_frame):
        """Capture a calibration image pair"""
        # Convert to grayscale
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, self.checkerboard_size, None
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, self.checkerboard_size, None
        )
        
        if ret_left and ret_right:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            # Store images and corners
            self.left_images.append((gray_left, corners_left))
            self.right_images.append((gray_right, corners_right))
            
            # Draw corners for visualization
            vis_left = cv2.drawChessboardCorners(left_frame.copy(), self.checkerboard_size, corners_left, ret_left)
            vis_right = cv2.drawChessboardCorners(right_frame.copy(), self.checkerboard_size, corners_right, ret_right)
            
            return True, np.hstack([vis_left, vis_right])
        
        return False, np.hstack([left_frame, right_frame])
    
    def calibrate(self):
        """Run stereo calibration"""
        if len(self.left_images) < 10:
            print(f"Need at least 10 image pairs, have {len(self.left_images)}")
            return False
            
        print(f"Calibrating with {len(self.left_images)} image pairs...")
        
        # Prepare object points
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        objpoints = [objp for _ in range(len(self.left_images))]
        imgpoints_left = [corners for _, corners in self.left_images]
        imgpoints_right = [corners for _, corners in self.right_images]
        
        h, w = self.left_images[0][0].shape
        
        # Calibrate individual cameras
        print("Calibrating left camera...")
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objpoints, imgpoints_left, (w, h), None, None
        )
        
        print("Calibrating right camera...")
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objpoints, imgpoints_right, (w, h), None, None
        )
        
        # Stereo calibration
        print("Stereo calibration...")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        
        ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtx_left, dist_left, mtx_right, dist_right,
            (w, h), criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        print(f"Stereo calibration RMS error: {ret_stereo:.3f}")
        
        # Compute rectification transforms
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right,
            (w, h), R, T, alpha=0
        )
        
        # Store calibration data
        self.calibration_data.update({
            'timestamp': datetime.now().isoformat(),
            'image_size': [w, h],
            'rms_error': ret_stereo,
            'left_camera': {
                'matrix': mtx_left.tolist(),
                'distortion': dist_left.tolist(),
                'rectification': R1.tolist(),
                'projection': P1.tolist(),
                'roi': roi_left
            },
            'right_camera': {
                'matrix': mtx_right.tolist(),
                'distortion': dist_right.tolist(),
                'rectification': R2.tolist(),
                'projection': P2.tolist(),
                'roi': roi_right
            },
            'stereo': {
                'rotation': R.tolist(),
                'translation': T.tolist(),
                'essential': E.tolist(),
                'fundamental': F.tolist(),
                'disparity_to_depth': Q.tolist()
            }
        })
        
        # Compute baseline from translation vector
        baseline_calculated = np.linalg.norm(T)
        print(f"Calculated baseline: {baseline_calculated:.1f}mm (expected: {self.calibration_data['baseline_mm']}mm)")
        
        # Detect misalignment
        rotation_angle = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi
        print(f"Rotation misalignment: {rotation_angle:.2f} degrees")
        
        vertical_offset = T[1]
        print(f"Vertical offset: {vertical_offset:.1f}mm")
        
        return True
    
    def save_calibration(self, filename="stereo_calibration.json"):
        """Save calibration to file"""
        with open(filename, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        print(f"Calibration saved to {filename}")
        
    def load_calibration(self, filename="stereo_calibration.json"):
        """Load calibration from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"Calibration loaded from {filename}")
            return True
        return False
    
    def create_rectification_maps(self):
        """Create rectification maps for real-time processing"""
        if not self.calibration_data.get('left_camera'):
            print("No calibration data available")
            return None
            
        w, h = self.calibration_data['image_size']
        
        # Left camera
        mtx_left = np.array(self.calibration_data['left_camera']['matrix'])
        dist_left = np.array(self.calibration_data['left_camera']['distortion'])
        R1 = np.array(self.calibration_data['left_camera']['rectification'])
        P1 = np.array(self.calibration_data['left_camera']['projection'])
        
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, (w, h), cv2.CV_16SC2
        )
        
        # Right camera
        mtx_right = np.array(self.calibration_data['right_camera']['matrix'])
        dist_right = np.array(self.calibration_data['right_camera']['distortion'])
        R2 = np.array(self.calibration_data['right_camera']['rectification'])
        P2 = np.array(self.calibration_data['right_camera']['projection'])
        
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, (w, h), cv2.CV_16SC2
        )
        
        return {
            'left': (map1_left, map2_left),
            'right': (map1_right, map2_right)
        }
    
    def rectify_pair(self, left_frame, right_frame, maps):
        """Apply rectification to image pair"""
        rect_left = cv2.remap(left_frame, maps['left'][0], maps['left'][1], cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_frame, maps['right'][0], maps['right'][1], cv2.INTER_LINEAR)
        return rect_left, rect_right


def run_calibration_capture():
    """Interactive calibration capture tool"""
    print("\n📐 Stereo Camera Calibration")
    print("="*40)
    print("1. Print a 9x6 checkerboard pattern")
    print("2. Hold it at various positions and angles")
    print("3. Press SPACE to capture calibration image")
    print("4. Press 'c' to run calibration")
    print("5. Press 'q' to quit")
    
    calibrator = StereoCalibrator()
    
    # Camera pipelines
    def get_pipeline(sensor_id):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1"
        )
    
    cap_left = cv2.VideoCapture(get_pipeline(0), cv2.CAP_GSTREAMER)
    cap_right = cv2.VideoCapture(get_pipeline(1), cv2.CAP_GSTREAMER)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Failed to open cameras")
        return
        
    capture_count = 0
    
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if ret_left and ret_right:
            # Try to find checkerboard
            found, vis = calibrator.capture_calibration_pair(frame_left, frame_right)
            
            # Add status text
            status = "FOUND" if found else "No checkerboard"
            color = (0, 255, 0) if found else (0, 0, 255)
            cv2.putText(vis, f"Status: {status} | Captured: {capture_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Calibration Capture', vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and found:
                capture_count = len(calibrator.left_images)
                print(f"Captured calibration pair #{capture_count}")
            elif key == ord('c'):
                if calibrator.calibrate():
                    calibrator.save_calibration()
                    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_calibration_capture()