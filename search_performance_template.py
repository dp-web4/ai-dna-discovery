#!/usr/bin/env python3
"""
Search Performance Results Template
Customize the search parameters for your domain
"""

import argparse
from performance_tracker import PerformanceTracker
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Search performance test results")
    
    # Basic search options (keep these)
    parser.add_argument("--file", "-f", help="Search by filename (partial match)")
    parser.add_argument("--type", "-t", help="Filter by test type")
    parser.add_argument("--who", "-w", help="Filter by who ran (user, claude, automated)")
    parser.add_argument("--days", type=int, default=30, help="Days to look back (default: 30)")
    parser.add_argument("--success-only", action="store_true", help="Show only successful tests")
    parser.add_argument("--failed-only", action="store_true", help="Show only failed tests")
    parser.add_argument("--details", "-d", type=int, help="Show full details for test ID")
    parser.add_argument("--summary", "-s", action="store_true", help="Show summary table")
    parser.add_argument("--export", "-e", help="Export results to JSON file")
    
    # CUSTOMIZE: Add domain-specific search options
    # Vision example:
    # parser.add_argument("--min-fps", type=float, help="Minimum FPS")
    # parser.add_argument("--gpu", action="store_true", help="GPU tests only")
    
    # LLM example:
    # parser.add_argument("--model", help="Filter by model name")
    # parser.add_argument("--min-tokens", type=float, help="Minimum tokens/sec")
    
    # Hardware example:
    # parser.add_argument("--min-efficiency", type=float, help="Minimum efficiency %")
    
    args = parser.parse_args()
    
    tracker = PerformanceTracker()
    
    # Handle detailed view
    if args.details:
        details = tracker.get_test_details(args.details)
        if details:
            print(f"\nTest Run #{details['id']}")
            print("="*50)
            
            # Standard fields
            print(f"Timestamp: {details['timestamp']}")
            print(f"File: {details['filename']}")
            print(f"Type: {details['test_type']}")
            print(f"Who ran: {details['who_ran']}")
            print(f"Success: {'✓ Yes' if details['success'] else '✗ No'}")
            
            # CUSTOMIZE: Display domain-specific fields
            # Add your domain's primary metrics here
            
            if details['notes']:
                print(f"Notes: {details['notes']}")
            if details['error_message']:
                print(f"Error: {details['error_message']}")
            
            # Additional metrics
            if details['additional_metrics']:
                print("\nAdditional Metrics:")
                for metric in details['additional_metrics']:
                    print(f"  {metric['metric_name']}: {metric['metric_value']} {metric['unit']}")
                    
            # System info
            if details['system_info']:
                info = json.loads(details['system_info'])
                print("\nSystem Info:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
        else:
            print(f"Test run #{args.details} not found")
        return
    
    # Build search parameters
    search_params = {
        'filename': args.file,
        'test_type': args.type,
        'who_ran': args.who,
        'days_back': args.days
    }
    
    # Handle success/failed filters
    if args.success_only:
        search_params['success_only'] = True
    
    # CUSTOMIZE: Add your domain-specific search parameters
    # Example: search_params['min_fps'] = args.min_fps
    
    # Search
    results = tracker.search_tests(**search_params)
    
    # Filter failed tests if requested
    if args.failed_only:
        results = [r for r in results if not r['success']]
    
    # Export if requested
    if args.export:
        with open(args.export, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Exported {len(results)} results to {args.export}")
        return
    
    # Display results
    if args.summary or not results:
        tracker.print_summary()
        if not results and any([args.file, args.type, args.who]):
            print("No results found for your search criteria")
    else:
        print(f"\nFound {len(results)} test(s)")
        print("="*100)
        
        for test in results[:50]:  # Limit display
            print(f"\n[{test['id']}] {test['timestamp']} - {test['filename']}")
            print(f"    Type: {test['test_type']} | Who: {test['who_ran']} | " +
                  f"Success: {'✓' if test['success'] else '✗'}")
            
            # CUSTOMIZE: Display your domain's key metrics
            # Vision example:
            # if test.get('fps_avg'):
            #     print(f"    FPS: {test['fps_avg']:.1f}")
            
            if test['notes']:
                print(f"    Notes: {test['notes']}")
            if test['error_message']:
                print(f"    Error: {test['error_message']}")
                
        if len(results) > 50:
            print(f"\n... and {len(results) - 50} more results")
            
        print(f"\nUse --details <id> to see full metrics for a specific test")
        print(f"Use --export <filename.json> to export all results")

if __name__ == "__main__":
    main()