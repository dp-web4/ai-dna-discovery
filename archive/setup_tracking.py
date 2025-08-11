#!/usr/bin/env python3
"""
Setup performance tracking for a new experiment area
"""

import os
import shutil
import sys
import argparse

def setup_tracking(experiment_dir, domain_type=None):
    """Setup performance tracking in a new experiment directory"""
    
    # Get the template directory
    template_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Files to copy
    files_to_copy = [
        ("performance_tracking_template.py", "performance_tracker.py"),
        ("search_performance_template.py", "search_performance.py"),
        ("record_test_template.py", "record_test.py")
    ]
    
    print(f"Setting up performance tracking in: {experiment_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Copy template files
    for src_name, dst_name in files_to_copy:
        src_path = os.path.join(template_dir, src_name)
        dst_path = os.path.join(experiment_dir, dst_name)
        
        if os.path.exists(dst_path):
            response = input(f"{dst_name} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print(f"Skipping {dst_name}")
                continue
                
        shutil.copy2(src_path, dst_path)
        print(f"✓ Copied {dst_name}")
    
    # Create a customization guide
    guide_path = os.path.join(experiment_dir, "CUSTOMIZE_TRACKING.md")
    
    guide_content = f"""# Customizing Performance Tracking for {os.path.basename(experiment_dir)}

## Quick Customization Steps

1. **Edit performance_tracker.py**:
   - Update the CREATE TABLE schema with your metrics
   - Modify the system_info dictionary
   - Add initialization examples in main()

2. **Update search_performance.py**:
   - Add domain-specific command line arguments
   - Customize the display format for your metrics

3. **Modify record_test.py**:
   - Add arguments for your specific metrics
   - Update the examples in the help text

## Domain-Specific Examples

"""
    
    if domain_type == "vision":
        guide_content += """### Vision/Graphics Metrics
- fps_avg, fps_min, fps_max
- processing_time_ms
- resolution
- gpu_used, gpu_library
- algorithm_name
"""
    elif domain_type == "llm":
        guide_content += """### LLM/AI Metrics
- model_name, model_size
- tokens_per_second
- memory_usage_mb
- accuracy, perplexity
- latency_ms
- prompt_length, response_length
"""
    elif domain_type == "hardware":
        guide_content += """### Hardware/Embedded Metrics
- voltage, current, power
- efficiency_percent
- temperature_c
- duty_cycle
- frequency_hz
"""
    else:
        guide_content += """### Common Metrics to Consider
- execution_time
- memory_usage
- accuracy/error_rate
- throughput
- resource_utilization
"""
    
    guide_content += """
## Testing Your Setup

1. Initialize the database:
   ```bash
   python3 performance_tracker.py
   ```

2. Record a test result:
   ```bash
   python3 record_test.py test_script.py --type benchmark --notes "First test"
   ```

3. Search results:
   ```bash
   python3 search_performance.py --summary
   ```

## Next Steps
- Delete this file once customization is complete
- Start recording all test results immediately
- Review results weekly to identify trends
"""
    
    with open(guide_path, 'w') as f:
        f.write(guide_content)
    print(f"✓ Created customization guide: CUSTOMIZE_TRACKING.md")
    
    # Make scripts executable
    for _, dst_name in files_to_copy:
        dst_path = os.path.join(experiment_dir, dst_name)
        if os.path.exists(dst_path):
            os.chmod(dst_path, 0o755)
    
    print(f"\n✅ Performance tracking setup complete!")
    print(f"\nNext steps:")
    print(f"1. cd {experiment_dir}")
    print(f"2. Review CUSTOMIZE_TRACKING.md")
    print(f"3. Edit performance_tracker.py for your metrics")
    print(f"4. Run: python3 performance_tracker.py")

def main():
    parser = argparse.ArgumentParser(
        description="Setup performance tracking for a new experiment area"
    )
    parser.add_argument("directory", help="Experiment directory path")
    parser.add_argument("--type", choices=["vision", "llm", "hardware", "general"],
                       help="Domain type for customization hints")
    
    args = parser.parse_args()
    
    # Convert to absolute path
    experiment_dir = os.path.abspath(args.directory)
    
    setup_tracking(experiment_dir, args.type)

if __name__ == "__main__":
    main()