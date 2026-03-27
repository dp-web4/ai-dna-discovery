#!/bin/bash
# Simple camera preview using GStreamer
# Press Ctrl+C to stop

echo "🎥 Starting camera preview..."
echo "Press Ctrl+C to stop"
echo ""
echo "Available preview modes:"
echo "1. 1920x1080 @ 30fps (default)"
echo "2. 1280x720 @ 60fps"
echo "3. 3280x2464 @ 21fps (full resolution)"
echo ""

# Default to 1080p
WIDTH=1920
HEIGHT=1080
FPS=30

# Check for command line argument
if [ "$1" = "2" ]; then
    WIDTH=1280
    HEIGHT=720
    FPS=60
    echo "Using 720p @ 60fps mode"
elif [ "$1" = "3" ]; then
    WIDTH=3280
    HEIGHT=2464
    FPS=21
    echo "Using full resolution mode"
else
    echo "Using 1080p @ 30fps mode (default)"
fi

# Run GStreamer pipeline with preview window
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
    "video/x-raw(memory:NVMM),width=$WIDTH,height=$HEIGHT,framerate=$FPS/1" ! \
    nvvidconv ! \
    "video/x-raw,width=960,height=540" ! \
    xvimagesink sync=false

echo ""
echo "Camera preview stopped"