# Testing Guidelines Advisory

*Last Updated: 2025-01-24*

## Testing Philosophy

1. **Test Early**: Bugs are cheaper to fix when fresh
2. **Test Often**: Small, frequent tests catch issues
3. **Test Thoroughly**: Edge cases matter
4. **Test Realistically**: Use production-like data
5. **Test Automatically**: Manual testing doesn't scale

## Types of Testing

### Unit Testing
Tests individual functions/methods in isolation
```python
def test_calculate_consciousness_score():
    # Test normal case
    score = calculate_consciousness_score(0.8, 0.9)
    assert 0.0 <= score <= 1.0
    
    # Test edge cases
    assert calculate_consciousness_score(0.0, 0.0) == 0.0
    assert calculate_consciousness_score(1.0, 1.0) == 1.0
    
    # Test invalid input
    with pytest.raises(ValueError):
        calculate_consciousness_score(-0.1, 0.5)
```

### Integration Testing
Tests how components work together
```python
def test_model_pipeline():
    # Test full pipeline
    data = load_test_data()
    processed = preprocess(data)
    model = load_model("test_model")
    predictions = model.predict(processed)
    
    assert predictions.shape == (len(data), num_classes)
    assert all(0 <= p <= 1 for p in predictions.flatten())
```

### End-to-End Testing
Tests complete workflows
```python
def test_experiment_workflow():
    # Run complete experiment
    config = load_config("test_config.yaml")
    experiment = Experiment(config)
    results = experiment.run()
    
    # Verify results structure
    assert "metrics" in results
    assert "visualizations" in results
    assert results["metrics"]["accuracy"] > 0.5
```

## Testing Best Practices

### Test Structure
```python
# Arrange - Act - Assert pattern
def test_function_name():
    # Arrange: Set up test conditions
    input_data = create_test_data()
    expected_output = known_good_result()
    
    # Act: Execute the function
    actual_output = function_under_test(input_data)
    
    # Assert: Verify results
    assert actual_output == expected_output
```

### Test Naming
```python
# Pattern: test_[function]_[condition]_[expected_result]
def test_load_model_valid_path_returns_model():
    pass

def test_load_model_missing_file_raises_error():
    pass

def test_process_data_empty_input_returns_empty():
    pass
```

### Test Data Management
```python
# Use fixtures for reusable test data
@pytest.fixture
def sample_dataset():
    return {
        "train": generate_samples(100),
        "test": generate_samples(20)
    }

def test_training(sample_dataset):
    model = train_model(sample_dataset["train"])
    assert model is not None
```

## AI/ML Specific Testing

### Model Testing
```python
def test_model_properties():
    model = create_model()
    
    # Test architecture
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1000)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 1000)
    
    # Test parameter count
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 100_000_000  # Less than 100M params
```

### Training Testing
```python
def test_training_convergence():
    # Use small dataset for fast testing
    small_data = create_minimal_dataset()
    model = create_simple_model()
    
    initial_loss = evaluate_model(model, small_data)
    trained_model = train_model(model, small_data, epochs=10)
    final_loss = evaluate_model(trained_model, small_data)
    
    # Should improve
    assert final_loss < initial_loss * 0.8
```

### Data Pipeline Testing
```python
def test_data_augmentation():
    original = load_image("test.jpg")
    augmented = augment_image(original)
    
    # Same shape, different content
    assert original.shape == augmented.shape
    assert not np.array_equal(original, augmented)
    
    # Stays in valid range
    assert augmented.min() >= 0
    assert augmented.max() <= 255
```

## Hardware and Performance Testing

### GPU Testing
```python
def test_gpu_availability():
    if torch.cuda.is_available():
        # Test GPU operations
        device = torch.device("cuda")
        tensor = torch.randn(1000, 1000).to(device)
        result = torch.matmul(tensor, tensor)
        assert result.device.type == "cuda"
    else:
        pytest.skip("GPU not available")
```

### Memory Testing
```python
def test_memory_usage():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run memory-intensive operation
    large_model = create_large_model()
    process_large_dataset(large_model)
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory
    
    # Should not leak excessively
    assert memory_increase < 1000  # Less than 1GB increase
```

### Performance Testing
```python
def test_inference_speed():
    model = load_model("production_model")
    test_input = create_test_batch(batch_size=32)
    
    # Warmup
    for _ in range(10):
        model(test_input)
    
    # Time inference
    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        model(test_input)
    
    avg_time = (time.time() - start_time) / iterations
    assert avg_time < 0.1  # Less than 100ms per batch
```

## Error Handling Testing

### Exception Testing
```python
def test_error_handling():
    # Test specific exceptions
    with pytest.raises(ValueError, match="Invalid input shape"):
        process_invalid_data([1, 2, 3])
    
    # Test error recovery
    try:
        risky_operation()
    except ExpectedException as e:
        assert cleanup_was_called()
        assert str(e) == "Expected error message"
```

### Edge Case Testing
```python
def test_edge_cases():
    # Empty input
    assert process_data([]) == []
    
    # Single element
    assert process_data([1]) == [1]
    
    # Large input
    large_input = list(range(1_000_000))
    result = process_data(large_input)
    assert len(result) == len(large_input)
    
    # Special values
    assert handle_special(float('inf')) == "infinity"
    assert handle_special(float('nan')) == "not a number"
```

## Test Environment Setup

### Requirements
```txt
# requirements-test.txt
pytest>=7.0.0
pytest-cov>=3.0.0
pytest-xdist>=2.5.0  # Parallel testing
pytest-timeout>=2.1.0
pytest-mock>=3.6.0
hypothesis>=6.0.0  # Property-based testing
```

### Configuration
```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    -x  # Stop on first failure
markers =
    slow: marks tests as slow
    gpu: marks tests requiring GPU
```

## Continuous Testing

### Pre-commit Testing
```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: tests
        entry: pytest tests/unit -x
        language: system
        pass_filenames: false
        always_run: true
```

### Test Automation
```yaml
# GitHub Actions example
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements-test.txt
          pytest tests/
```

## Common Testing Pitfalls

### Don't Test Implementation
```python
# Bad - tests implementation details
def test_internal_method():
    obj = MyClass()
    assert obj._internal_state == expected  # Don't test private

# Good - tests behavior
def test_public_behavior():
    obj = MyClass()
    result = obj.public_method()
    assert result == expected_output
```

### Don't Test External Dependencies
```python
# Bad - tests external service
def test_api_call():
    response = requests.get("https://api.example.com")
    assert response.status_code == 200  # Flaky!

# Good - mock external dependencies
def test_api_call(mock_requests):
    mock_requests.get.return_value.status_code = 200
    result = my_api_function()
    assert result == expected
```

### Don't Ignore Warnings
```python
# Capture and verify warnings
def test_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        old_function()
```

## Test Documentation

### Document Test Purpose
```python
def test_consciousness_emergence():
    """
    Test that consciousness score increases with network depth.
    
    This validates our hypothesis that deeper networks exhibit
    more emergent consciousness-like properties.
    """
    shallow = create_model(layers=2)
    deep = create_model(layers=10)
    
    shallow_score = measure_consciousness(shallow)
    deep_score = measure_consciousness(deep)
    
    assert deep_score > shallow_score
```

## Quick Testing Checklist

Before committing:
- [ ] All tests pass locally
- [ ] New features have tests
- [ ] Edge cases are covered
- [ ] Error paths are tested
- [ ] No hardcoded paths/values
- [ ] Tests run in isolation
- [ ] Performance is acceptable
- [ ] Documentation is updated

---

**Remember**: Tests are not overhead - they're insurance. The time spent writing tests is paid back many times over in prevented bugs and confident refactoring.