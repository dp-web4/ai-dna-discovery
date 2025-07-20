#!/usr/bin/env python3
"""
Test camera access and basic functionality
First step in sensor integration
"""

import cv2
import numpy as np
import time
import platform
import sys

def test_camera_availability():
    """Check which cameras are available"""
    print("🎥 Testing camera availability...")
    print(f"Platform: {platform.system()}")
    print(f"OpenCV version: {cv2.__version__}")
    
    available_cameras = []
    
    # Test up to 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"\n✅ Camera {i} found:")
            print(f"   Resolution: {int(width)}x{int(height)}")
            print(f"   FPS: {fps}")
            
            available_cameras.append({
                'index': i,
                'width': int(width),
                'height': int(height),
                'fps': fps
            })
            
            cap.release()
        
    if not available_cameras:
        print("\n❌ No cameras found!")
        print("\nTroubleshooting:")
        print("1. Check if cameras are connected")
        print("2. On WSL2, you may need USB passthrough")
        print("3. Try: sudo apt-get install v4l-utils")
        print("4. Check with: v4l2-ctl --list-devices")
    
    return available_cameras

def test_camera_capture(camera_index=0, duration=5):
    """Test capturing frames from camera"""
    print(f"\n📸 Testing camera {camera_index} capture for {duration} seconds...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        return False
    
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    start_time = time.time()
    frame_count = 0
    
    print("Press 'q' to quit early...")
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        frame_count += 1
        
        # Add frame info overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {time.time() - start_time:.1f}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(f'Camera {camera_index} Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User quit early")
            break
    
    # Calculate statistics
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n📊 Capture Statistics:")
    print(f"   Frames captured: {frame_count}")
    print(f"   Duration: {elapsed:.2f}s")
    print(f"   Actual FPS: {actual_fps:.2f}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    return True

def test_motion_detection(camera_index=0, duration=10):
    """Test basic motion detection"""
    print(f"\n🏃 Testing motion detection on camera {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        return False
    
    # Create background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2()
    
    start_time = time.time()
    motion_events = 0
    
    print("Move in front of camera to test motion detection...")
    print("Press 'q' to quit...")
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Apply background subtraction
        fgMask = backSub.apply(frame)
        
        # Count non-zero pixels (motion)
        motion_pixels = cv2.countNonZero(fgMask)
        motion_threshold = 1000  # Adjust based on your needs
        
        if motion_pixels > motion_threshold:
            motion_events += 1
            cv2.putText(frame, "MOTION DETECTED!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show both original and motion mask
        cv2.imshow('Camera Feed', frame)
        cv2.imshow('Motion Mask', fgMask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"\n📊 Motion Detection Results:")
    print(f"   Motion events: {motion_events}")
    print(f"   Duration: {time.time() - start_time:.2f}s")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return True

def test_stereo_vision(camera_0=0, camera_1=1):
    """Test if we can use both cameras for stereo vision"""
    print(f"\n👀 Testing stereo vision with cameras {camera_0} and {camera_1}...")
    
    cap0 = cv2.VideoCapture(camera_0)
    cap1 = cv2.VideoCapture(camera_1)
    
    if not cap0.isOpened() or not cap1.isOpened():
        print("❌ Cannot open both cameras for stereo vision")
        if cap0.isOpened():
            cap0.release()
        if cap1.isOpened():
            cap1.release()
        return False
    
    print("✅ Both cameras opened successfully")
    print("Press 'q' to quit...")
    
    for i in range(100):  # Test 100 frames
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print("❌ Failed to grab frames from both cameras")
            break
        
        # Resize frames to same size if needed
        frame0 = cv2.resize(frame0, (640, 480))
        frame1 = cv2.resize(frame1, (640, 480))
        
        # Combine frames side by side
        stereo_frame = np.hstack([frame0, frame1])
        
        cv2.putText(stereo_frame, "Camera 0", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(stereo_frame, "Camera 1", (650, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Stereo Vision Test', stereo_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    
    print("✅ Stereo vision test completed")
    return True

def main():
    print("=== Camera System Test ===\n")
    
    # Test 1: Check available cameras
    cameras = test_camera_availability()
    
    if not cameras:
        print("\n❌ No cameras available. Exiting.")
        return
    
    # Test 2: Capture from first camera
    print("\n" + "="*50)
    if test_camera_capture(cameras[0]['index'], duration=5):
        print("✅ Camera capture test passed")
    
    # Test 3: Motion detection
    print("\n" + "="*50)
    if test_motion_detection(cameras[0]['index'], duration=10):
        print("✅ Motion detection test passed")
    
    # Test 4: Stereo vision if we have 2 cameras
    if len(cameras) >= 2:
        print("\n" + "="*50)
        if test_stereo_vision(cameras[0]['index'], cameras[1]['index']):
            print("✅ Stereo vision test passed")
    else:
        print("\n⚠️  Only one camera found, skipping stereo vision test")
    
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    main()