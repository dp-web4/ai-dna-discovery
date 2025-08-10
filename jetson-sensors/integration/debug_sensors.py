#!/usr/bin/env python3
"""
Debug sensor connectivity and functionality
"""

import cv2
import serial
import numpy as np
import time
import sys

def test_cameras():
    """Test camera connectivity"""
    print("\n" + "="*50)
    print("TESTING CAMERAS")
    print("="*50)
    
    results = {}
    
    for cam_id in [0, 1]:
        print(f"\nTesting Camera {cam_id}...")
        try:
            cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
            
            # Try different resolutions
            resolutions = [
                (640, 480),
                (1280, 720),
                (1920, 1080),
                (3280, 2464)  # Native for some CSI cameras
            ]
            
            for width, height in resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    actual_h, actual_w = frame.shape[:2]
                    print(f"  ✓ Resolution: {actual_w}x{actual_h}")
                    
                    # Save test frame
                    filename = f"test_cam{cam_id}_{actual_w}x{actual_h}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"    Saved: {filename}")
                    
                    if cam_id not in results:
                        results[cam_id] = {
                            'status': 'OK',
                            'resolution': (actual_w, actual_h),
                            'frame': frame
                        }
                    break
            
            cap.release()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[cam_id] = {'status': 'ERROR', 'error': str(e)}
    
    return results

def test_imu():
    """Test IMU connectivity"""
    print("\n" + "="*50)
    print("TESTING IMU")
    print("="*50)
    
    ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0']
    baudrates = [115200, 9600]
    
    for port in ports:
        for baudrate in baudrates:
            print(f"\nTrying {port} at {baudrate} baud...")
            try:
                ser = serial.Serial(
                    port=port,
                    baudrate=baudrate,
                    timeout=1.0,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                
                # Try to read some data
                time.sleep(0.5)
                if ser.in_waiting > 0:
                    data = ser.read(min(100, ser.in_waiting))
                    print(f"  ✓ Connected! Received {len(data)} bytes")
                    print(f"    First 20 bytes (hex): {data[:20].hex()}")
                    ser.close()
                    return {'status': 'OK', 'port': port, 'baudrate': baudrate}
                else:
                    # Try sending a query command (if supported)
                    ser.write(b'\r\n')
                    time.sleep(0.5)
                    if ser.in_waiting > 0:
                        data = ser.read(ser.in_waiting)
                        print(f"  ✓ Connected! Response: {len(data)} bytes")
                        ser.close()
                        return {'status': 'OK', 'port': port, 'baudrate': baudrate}
                
                ser.close()
                print(f"  - No data received")
                
            except Exception as e:
                if "Permission denied" in str(e):
                    print(f"  ✗ Permission denied (may need sudo or user in dialout group)")
                elif "No such file" in str(e):
                    print(f"  - Port doesn't exist")
                else:
                    print(f"  ✗ Error: {e}")
    
    return {'status': 'NOT_FOUND'}

def test_audio():
    """Test audio without PyAudio complications"""
    print("\n" + "="*50)
    print("TESTING AUDIO")
    print("="*50)
    
    import subprocess
    
    # Check ALSA devices
    print("\nALSA Playback Devices:")
    try:
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"  Error listing playback devices: {e}")
    
    print("\nALSA Capture Devices:")
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"  Error listing capture devices: {e}")
    
    # Check Bluetooth
    print("\nBluetooth Devices:")
    try:
        result = subprocess.run(['bluetoothctl', 'devices'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"  Error listing Bluetooth devices: {e}")
    
    return {'status': 'CHECK_OUTPUT'}

def test_simple_fusion():
    """Test simple sensor fusion without audio"""
    print("\n" + "="*50)
    print("SIMPLE SENSOR FUSION TEST")
    print("="*50)
    
    cam_results = test_cameras()
    
    if not any(r.get('status') == 'OK' for r in cam_results.values()):
        print("\n✗ No cameras available for fusion test")
        return
    
    print("\nStarting simple fusion display (press 'q' to quit)...")
    
    # Create simple display canvas
    canvas_width = 1280
    canvas_height = 720
    
    # Get available camera
    cam_id = None
    for cid, result in cam_results.items():
        if result.get('status') == 'OK':
            cam_id = cid
            break
    
    if cam_id is None:
        print("No working camera found")
        return
    
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    
    frame_count = 0
    start_time = time.time()
    
    print("\nCapturing frames...")
    
    for i in range(100):  # Capture 100 frames
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Resize frame for display
        frame = cv2.resize(frame, (640, 480))
        
        # Create canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
        
        # Place camera feed
        canvas[50:530, 50:690] = frame
        
        # Add text overlays
        cv2.putText(canvas, f"CAMERA {cam_id}", (50, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add FPS counter
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            cv2.putText(canvas, f"FPS: {fps:.1f}", (50, 560),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add simple motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        motion_pixels = np.sum(edges > 0)
        motion_percent = (motion_pixels / (640 * 480)) * 100
        
        cv2.putText(canvas, f"Motion: {motion_percent:.1f}%", (50, 590),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Save every 10th frame
        if frame_count % 10 == 0:
            filename = f"fusion_frame_{frame_count:04d}.jpg"
            cv2.imwrite(filename, canvas)
            print(f"  Saved {filename}")
        
        # Small delay
        time.sleep(0.03)
    
    cap.release()
    
    print(f"\nTest complete. Captured {frame_count} frames in {elapsed:.1f} seconds")
    print(f"Average FPS: {frame_count/elapsed:.1f}")

def main():
    """Run all sensor tests"""
    print("\n" + "█"*50)
    print(" "*15 + "SENSOR DEBUG UTILITY")
    print("█"*50)
    
    # Test each component
    cam_results = test_cameras()
    imu_result = test_imu()
    audio_result = test_audio()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    # Camera summary
    for cam_id, result in cam_results.items():
        if result.get('status') == 'OK':
            res = result.get('resolution', (0, 0))
            print(f"Camera {cam_id}: ✓ OK ({res[0]}x{res[1]})")
        else:
            print(f"Camera {cam_id}: ✗ {result.get('error', 'Unknown error')}")
    
    # IMU summary
    if imu_result.get('status') == 'OK':
        print(f"IMU: ✓ OK ({imu_result['port']} @ {imu_result['baudrate']} baud)")
    else:
        print(f"IMU: ✗ Not found")
    
    # Audio summary
    print(f"Audio: Check output above for device list")
    
    # Run simple fusion test if cameras work
    if any(r.get('status') == 'OK' for r in cam_results.values()):
        print("\n" + "="*50)
        response = input("Run simple fusion test? (y/n): ")
        if response.lower() == 'y':
            test_simple_fusion()
    
    print("\n" + "█"*50)
    print(" "*18 + "DEBUG COMPLETE")
    print("█"*50 + "\n")

if __name__ == "__main__":
    main()