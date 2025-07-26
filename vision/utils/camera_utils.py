#!/usr/bin/env python3
"""
Camera utilities for Jetson with proper GStreamer pipelines
"""

import cv2

def get_jetson_gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0
):
    """
    Return GStreamer pipeline for CSI camera on Jetson
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1"
    )

def open_camera(width=1280, height=720, fps=30):
    """
    Open camera with appropriate backend for platform
    """
    # Try Jetson CSI camera first
    gst_pipeline = get_jetson_gstreamer_pipeline(
        display_width=width,
        display_height=height,
        framerate=fps
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        print("✅ Opened CSI camera with GStreamer")
        return cap
    
    # Fallback to V4L2
    print("Trying V4L2 fallback...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("✅ Opened camera with V4L2")
        return cap
    
    return None