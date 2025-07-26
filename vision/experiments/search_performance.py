#!/usr/bin/env python3
"""
Search and query performance test results
"""

import argparse
from performance_tracker import PerformanceTracker
import json

def main():
    parser = argparse.ArgumentParser(description="Search performance test results")
    parser.add_argument("--file", "-f", help="Search by filename (partial match)")
    parser.add_argument("--type", "-t", help="Filter by test type (realtime, benchmark)")
    parser.add_argument("--gpu", action="store_true", help="Show only GPU tests")
    parser.add_argument("--cpu", action="store_true", help="Show only CPU tests")
    parser.add_argument("--min-fps", type=float, help="Minimum FPS")
    parser.add_argument("--days", type=int, default=30, help="Days to look back (default: 30)")
    parser.add_argument("--details", "-d", type=int, help="Show full details for test ID")
    parser.add_argument("--who", "-w", help="Filter by who ran (user, claude, automated)")
    parser.add_argument("--summary", "-s", action="store_true", help="Show summary table")
    
    args = parser.parse_args()
    
    tracker = PerformanceTracker()
    
    if args.details:
        # Show detailed info for specific test
        details = tracker.get_test_details(args.details)
        if details:
            print(f"\nTest Run #{details['id']}")
            print("="*50)
            print(f"Timestamp: {details['timestamp']}")
            print(f"File: {details['filename']}")
            print(f"Type: {details['test_type']}")
            print(f"Who ran: {details['who_ran']}")
            print(f"FPS: {details['fps_avg']:.1f}" if details['fps_avg'] else "FPS: N/A")
            if details['fps_min'] and details['fps_max']:
                print(f"FPS Range: {details['fps_min']:.1f} - {details['fps_max']:.1f}")
            if details['processing_time_ms']:
                print(f"Processing Time: {details['processing_time_ms']:.2f}ms")
            print(f"GPU Used: {'Yes' if details['gpu_used'] else 'No'}")
            if details['gpu_library']:
                print(f"GPU Library: {details['gpu_library']}")
            print(f"Resolution: {details['resolution']}")
            if details['notes']:
                print(f"Notes: {details['notes']}")
            
            if details['additional_metrics']:
                print("\nAdditional Metrics:")
                for metric in details['additional_metrics']:
                    print(f"  {metric['metric_name']}: {metric['metric_value']} {metric['unit']}")
        else:
            print(f"Test run #{args.details} not found")
        return
    
    # Search tests
    gpu_used = None
    if args.gpu:
        gpu_used = True
    elif args.cpu:
        gpu_used = False
        
    results = tracker.search_tests(
        filename=args.file,
        test_type=args.type,
        gpu_used=gpu_used,
        min_fps=args.min_fps,
        days_back=args.days
    )
    
    # Filter by who ran if specified
    if args.who:
        results = [r for r in results if r['who_ran'] == args.who]
    
    if args.summary or not results:
        tracker.print_summary()
        if not results and any([args.file, args.type, args.gpu, args.cpu, args.min_fps, args.who]):
            print("No results found for your search criteria")
    else:
        # Detailed results
        print(f"\nFound {len(results)} test(s)")
        print("="*100)
        
        for i, test in enumerate(results):
            print(f"\n[{test['id']}] {test['timestamp']} - {test['filename']}")
            print(f"    Type: {test['test_type']} | Who: {test['who_ran']}")
            if test['fps_avg']:
                print(f"    FPS: {test['fps_avg']:.1f}", end="")
                if test['fps_min'] and test['fps_max']:
                    print(f" (range: {test['fps_min']:.1f}-{test['fps_max']:.1f})")
                else:
                    print()
            if test['processing_time_ms']:
                print(f"    Processing: {test['processing_time_ms']:.2f}ms")
            print(f"    GPU: {'Yes' if test['gpu_used'] else 'No'}", end="")
            if test['gpu_library']:
                print(f" ({test['gpu_library']})")
            else:
                print()
            if test['notes']:
                print(f"    Notes: {test['notes']}")
                
        print("\nUse --details <id> to see full metrics for a specific test")

if __name__ == "__main__":
    main()