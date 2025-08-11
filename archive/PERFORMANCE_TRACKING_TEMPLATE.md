# Performance Tracking System Template

This template provides a standardized way to track performance and results across all AI DNA Discovery experiments.

## Quick Start for New Experiment Area

1. **Copy the tracking system to your experiment folder**:
   ```bash
   cd your_experiment_folder/
   cp ../../performance_tracking_template.py performance_tracker.py
   cp ../../search_performance_template.py search_performance.py
   cp ../../record_test_template.py record_test.py
   ```

2. **Customize the tracker for your domain** by editing `performance_tracker.py`:
   - Update the table schema for domain-specific metrics
   - Add relevant test types
   - Modify default values

3. **Initialize your database**:
   ```bash
   python3 performance_tracker.py
   ```

## Database Schema

### Core Fields (Always Include)
- **id**: Unique identifier
- **timestamp**: When the test was run
- **filename**: Script/notebook that was tested
- **test_type**: Type of test (varies by domain)
- **who_ran**: user, claude, or automated
- **success**: Boolean - did the test complete successfully?
- **notes**: Context and observations
- **system_info**: JSON field for environment details

### Performance Metrics (Customize per Domain)
For vision experiments:
- fps_avg, fps_min, fps_max
- processing_time_ms
- gpu_used, gpu_library

For LLM experiments:
- tokens_per_second
- memory_usage_mb
- model_name, model_size

For battery/hardware experiments:
- voltage, current, power
- efficiency_percent
- temperature_c

## Usage Patterns

### Recording Results
```bash
# Simple recording
python3 record_test.py experiment_name --success --metric1 value1 --notes "conditions"

# With multiple metrics
python3 record_test.py advanced_test.py \
  --success \
  --accuracy 0.95 \
  --latency 230 \
  --memory 512 \
  --notes "Optimized parameters, batch size 32"
```

### Searching Results
```bash
# View recent summary
python3 search_performance.py --summary

# Find specific tests
python3 search_performance.py --file "experiment" --days 7

# Filter by success
python3 search_performance.py --success-only

# Get detailed metrics
python3 search_performance.py --details 5
```

## Best Practices

### 1. Always Record Context
- Who ran it (user interaction vs automated test)
- Environmental conditions
- Configuration parameters
- Any anomalies observed

### 2. Use Consistent Naming
- Script names should be descriptive
- Test types should be standardized
- Metrics should have clear units

### 3. Immediate Recording
- Record results immediately after tests
- Don't rely on memory or scattered notes
- Include failed tests with error details

### 4. Regular Analysis
- Review trends weekly
- Compare similar tests
- Identify performance regressions

## Example Implementations

### Vision (Already Implemented)
- Location: `/vision/experiments/`
- Tracks: FPS, processing time, GPU usage
- Test types: realtime, benchmark, stress

### Future Examples

#### LLM Memory System
```python
tracker.record_test(
    filename="memory_rag_test.py",
    test_type="inference",
    who_ran="automated",
    success=True,
    notes="Testing RAG with 1000 documents",
    additional_metrics={
        "tokens_per_second": (45.2, "tokens/s"),
        "memory_usage": (1024, "MB"),
        "accuracy": (0.89, "ratio"),
        "retrieval_time": (123, "ms")
    }
)
```

#### Battery Management
```python
tracker.record_test(
    filename="cell_balancing_v2.py",
    test_type="balancing",
    who_ran="user",
    success=True,
    notes="4S2P configuration, 25C ambient",
    additional_metrics={
        "balance_time": (45, "minutes"),
        "efficiency": (94.5, "percent"),
        "max_cell_delta": (0.05, "V"),
        "temperature_rise": (3.2, "C")
    }
)
```

## Automation Support

### Git Hook Integration
Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Remind to record test results
echo "Remember to record any performance tests in the tracking database!"
```

### CI/CD Integration
```yaml
# In your CI pipeline
- name: Run Performance Tests
  run: |
    python3 run_benchmarks.py
    python3 record_test.py "CI Benchmark" --who automated \
      --fps ${{ env.BENCHMARK_FPS }} \
      --notes "CI run ${{ github.run_number }}"
```

## Directory Structure
```
experiment_area/
├── performance_tracker.py    # Domain-customized tracker
├── search_performance.py     # Search interface
├── record_test.py           # Recording utility
├── performance_tests.db     # SQLite database
└── PERFORMANCE_RESULTS.md   # Human-readable summary
```

## Migration and Backup
```bash
# Backup database
cp performance_tests.db performance_tests_$(date +%Y%m%d).db

# Export to CSV
sqlite3 performance_tests.db ".mode csv" ".headers on" \
  "SELECT * FROM test_runs" > results_export.csv
```

Remember: Accurate tracking enables:
- Performance optimization
- Regression detection  
- Knowledge preservation
- Reproducible research

Start tracking from day one in each new experiment area!