#!/usr/bin/env python3
"""
Fix excessive code formatting in COMPREHENSIVE_REPORT.md
Converts code blocks to proper prose, lists, and tables where appropriate
"""

import re

def fix_chapter_26(content):
    """Replace Chapter 26 with revised version"""
    # Read the revised chapter
    with open('CHAPTER_26_REVISED.md', 'r', encoding='utf-8') as f:
        revised_chapter = f.read()
    
    # Find Chapter 26 start and end
    chapter_start = content.find("## Chapter 26: Calls to Action")
    appendices_start = content.find("# Appendices")
    
    if chapter_start != -1 and appendices_start != -1:
        # Extract content before and after Chapter 26
        before = content[:chapter_start]
        after = content[appendices_start-5:]  # Include the --- separator
        
        # Combine with revised chapter
        content = before + revised_chapter + "\n\n---\n\n" + after
    
    return content

def fix_performance_tables(content):
    """Convert code block performance metrics to proper tables"""
    
    # Example: Fix GPU Utilization metrics around line 6545-6595
    gpu_metrics_pattern = r'```python\s*\n.*?gpu_utilization_evolution = \{[\s\S]*?\}\s*\n```'
    
    gpu_metrics_replacement = """### GPU Utilization Evolution

| Configuration | Initial | After Fix | Improvement |
|--------------|---------|-----------|-------------|
| RTX 4090 - Standard Trainer | 0% (memory only) | 95-98% | Complete fix |
| RTX 4090 - Custom Loop | N/A | 95-98% | Built for purpose |
| V100 (Colab) | 0% | 85-90% | Good utilization |
| T4 (Colab Free) | 0% | 70-80% | Acceptable |"""
    
    content = re.sub(gpu_metrics_pattern, gpu_metrics_replacement, content, flags=re.MULTILINE)
    
    return content

def fix_timeline_sections(content):
    """Convert code block timelines to proper timeline format"""
    
    # Fix implementation timeline around line 7498-7533
    timeline_pattern = r'```python\s*\n.*?implementation_timeline = \{[\s\S]*?\}\s*\n```'
    
    timeline_replacement = """### Implementation Timeline

**Phase 1: Foundation (Months 1-2)**
- Complete training for remaining 5 models
- Standardize training pipelines
- Create comprehensive documentation
- Set up automated testing

**Phase 2: Enhancement (Months 3-4)**
- Implement multi-model consensus
- Add real-time translation APIs
- Create web interface
- Deploy to cloud infrastructure

**Phase 3: Expansion (Months 5-6)**
- Launch developer SDK
- Create educational materials
- Build community platform
- Establish partnerships"""
    
    content = re.sub(timeline_pattern, timeline_replacement, content, flags=re.MULTILINE)
    
    return content

def fix_chapter_20_metrics(content):
    """Fix Chapter 20 performance metrics sections"""
    
    # Speed benchmarks pattern
    speed_pattern = r'```python\s*\nspeed_benchmarks = \{[\s\S]*?inference_times[\s\S]*?\}\s*\n```'
    
    speed_replacement = """### Speed Benchmarks Across Platforms

**Training Performance:**

| Platform | Model | Dataset Size | Time | Tokens/sec |
|----------|-------|--------------|------|------------|
| RTX 4090 | TinyLlama | 1,312 | 8 min | 487 |
| RTX 4090 | TinyLlama | 101 | 90 sec | 423 |
| V100 | TinyLlama | 101 | 3 min | 287 |
| T4 | TinyLlama | 101 | 7 min | 124 |

**Inference Performance:**

| Platform | Batch Size | Latency | Throughput |
|----------|------------|---------|------------|
| RTX 4090 | 8 | 12ms | 387 tok/s |
| Jetson Orin | 1 | 89ms | 45 tok/s |
| Raspberry Pi 4 | 1 | 423ms | 9 tok/s |
| CPU (i9) | 1 | 478ms | 8 tok/s |"""
    
    content = re.sub(speed_pattern, speed_replacement, content, flags=re.MULTILINE)
    
    return content

def main():
    # Read the report
    with open('COMPREHENSIVE_REPORT.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Fixing report formatting...")
    
    # Apply fixes
    content = fix_chapter_26(content)
    print("✓ Fixed Chapter 26")
    
    content = fix_performance_tables(content)
    print("✓ Fixed performance tables")
    
    content = fix_timeline_sections(content)
    print("✓ Fixed timeline sections")
    
    content = fix_chapter_20_metrics(content)
    print("✓ Fixed Chapter 20 metrics")
    
    # Write the fixed report
    with open('COMPREHENSIVE_REPORT_FIXED.md', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\nReport fixed and saved to COMPREHENSIVE_REPORT_FIXED.md")
    print("Review the changes before replacing the original.")

if __name__ == "__main__":
    main()