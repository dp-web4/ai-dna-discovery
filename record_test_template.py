#!/usr/bin/env python3
"""
Record Test Results Template
Customize this for your specific domain's metrics
"""

import argparse
from performance_tracker import PerformanceTracker
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Record a performance test result",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic recording
  python3 record_test.py my_test.py --type benchmark --success
  
  # With metrics and notes
  python3 record_test.py experiment.py --type realtime --who user \\
    --metric1 value1 --metric2 value2 --notes "Test conditions"
  
  # Failed test
  python3 record_test.py failed_test.py --type stress --failed \\
    --error "Out of memory at 1000 iterations"
"""
    )
    
    # Required arguments
    parser.add_argument("filename", help="Script/notebook that was tested")
    
    # Basic options (keep these)
    parser.add_argument("--type", "-t", required=True, 
                       help="Type of test (customize options for your domain)")
    parser.add_argument("--who", "-w", default="user", 
                       choices=["user", "claude", "automated"],
                       help="Who ran the test (default: user)")
    parser.add_argument("--success", dest="success", action="store_true", default=True,
                       help="Test completed successfully (default)")
    parser.add_argument("--failed", dest="success", action="store_false",
                       help="Test failed")
    parser.add_argument("--notes", "-n", help="Additional notes about the test")
    parser.add_argument("--error", "-e", help="Error message (if test failed)")
    
    # CUSTOMIZE: Add your domain-specific metrics as arguments
    # Vision example:
    # parser.add_argument("--fps", type=float, help="Average FPS")
    # parser.add_argument("--fps-min", type=float, help="Minimum FPS")
    # parser.add_argument("--fps-max", type=float, help="Maximum FPS")
    # parser.add_argument("--gpu", action="store_true", help="GPU was used")
    # parser.add_argument("--gpu-lib", help="GPU library (cupy, tensorrt, etc)")
    
    # LLM example:
    # parser.add_argument("--model", help="Model name")
    # parser.add_argument("--tokens", type=float, help="Tokens per second")
    # parser.add_argument("--memory", type=float, help="Memory usage in MB")
    # parser.add_argument("--accuracy", type=float, help="Accuracy score")
    
    # Hardware example:
    # parser.add_argument("--voltage", type=float, help="Voltage (V)")
    # parser.add_argument("--current", type=float, help="Current (A)")
    # parser.add_argument("--temp", type=float, help="Temperature (C)")
    # parser.add_argument("--efficiency", type=float, help="Efficiency (%)")
    
    # Generic additional metrics
    parser.add_argument("--metric", "-m", action="append", nargs=3,
                       metavar=("name", "value", "unit"),
                       help="Additional metric (can be used multiple times)")
    
    args = parser.parse_args()
    
    # Validate
    if args.error and args.success:
        print("Warning: Error message provided but test marked as success")
    
    if not args.success and not args.error:
        print("Warning: Test marked as failed but no error message provided")
        
    # Initialize tracker
    tracker = PerformanceTracker()
    
    # Prepare kwargs for domain-specific fields
    kwargs = {}
    
    # CUSTOMIZE: Add your domain-specific fields to kwargs
    # Example:
    # if hasattr(args, 'fps') and args.fps:
    #     kwargs['fps_avg'] = args.fps
    # if hasattr(args, 'gpu') and args.gpu:
    #     kwargs['gpu_used'] = True
    
    # Handle additional metrics
    additional_metrics = {}
    if args.metric:
        for name, value, unit in args.metric:
            try:
                additional_metrics[name] = (float(value), unit)
            except ValueError:
                print(f"Warning: Could not convert {value} to float for metric {name}")
    
    # Record the test
    try:
        test_id = tracker.record_test(
            filename=os.path.basename(args.filename),
            test_type=args.type,
            who_ran=args.who,
            success=args.success,
            notes=args.notes,
            error_message=args.error,
            additional_metrics=additional_metrics if additional_metrics else None,
            **kwargs
        )
        
        print(f"✅ Test recorded with ID: {test_id}")
        print(f"   File: {os.path.basename(args.filename)}")
        print(f"   Type: {args.type}")
        print(f"   Success: {'Yes' if args.success else 'No'}")
        
        # CUSTOMIZE: Display recorded metrics
        # if 'fps_avg' in kwargs:
        #     print(f"   FPS: {kwargs['fps_avg']}")
        
        if args.notes:
            print(f"   Notes: {args.notes}")
        if args.error:
            print(f"   Error: {args.error}")
            
        print(f"\nView details: python3 search_performance.py --details {test_id}")
        
    except Exception as e:
        print(f"❌ Error recording test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()