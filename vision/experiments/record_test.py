#!/usr/bin/env python3
"""
Record a new performance test result
"""

import argparse
from performance_tracker import PerformanceTracker
import os

def main():
    parser = argparse.ArgumentParser(description="Record a performance test result")
    parser.add_argument("filename", help="Script that was tested")
    parser.add_argument("--fps", type=float, required=True, help="Average FPS achieved")
    parser.add_argument("--type", default="realtime", choices=["realtime", "benchmark", "stress"],
                       help="Type of test (default: realtime)")
    parser.add_argument("--who", default="user", choices=["user", "claude", "automated"],
                       help="Who ran the test (default: user)")
    parser.add_argument("--gpu", action="store_true", help="GPU was used")
    parser.add_argument("--gpu-lib", help="GPU library used (cupy, vpi, tensorrt)")
    parser.add_argument("--fps-min", type=float, help="Minimum FPS")
    parser.add_argument("--fps-max", type=float, help="Maximum FPS")
    parser.add_argument("--time", type=float, help="Processing time in ms")
    parser.add_argument("--resolution", default="1280x720", help="Resolution (default: 1280x720)")
    parser.add_argument("--notes", help="Additional notes about the test")
    
    args = parser.parse_args()
    
    tracker = PerformanceTracker()
    
    # Ensure filename is just the basename
    filename = os.path.basename(args.filename)
    
    test_id = tracker.record_test(
        filename=filename,
        test_type=args.type,
        who_ran=args.who,
        fps_avg=args.fps,
        fps_min=args.fps_min,
        fps_max=args.fps_max,
        processing_time_ms=args.time,
        gpu_used=args.gpu,
        gpu_library=args.gpu_lib,
        resolution=args.resolution,
        notes=args.notes
    )
    
    print(f"✅ Test recorded with ID: {test_id}")
    print(f"   File: {filename}")
    print(f"   FPS: {args.fps}")
    print(f"   GPU: {'Yes' if args.gpu else 'No'}")
    if args.notes:
        print(f"   Notes: {args.notes}")

if __name__ == "__main__":
    main()