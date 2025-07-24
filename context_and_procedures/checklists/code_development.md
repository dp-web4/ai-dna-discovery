# Code Development Checklist

*Last Updated: 2025-01-24*

## Project Setup

### Initial Setup
1. [ ] Create project directory
2. [ ] Initialize git repository
3. [ ] Create `.gitignore` FIRST
4. [ ] Add standard patterns to `.gitignore`
5. [ ] Create README.md with project overview
6. [ ] Set up virtual environment (AFTER `.gitignore`)

### Virtual Environment Setup
```bash
# AFTER .gitignore is ready
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Dependency Management
1. [ ] Create requirements.txt
2. [ ] Pin specific versions
3. [ ] Document why each dependency is needed
4. [ ] Test fresh install regularly
5. [ ] Keep dependencies minimal

## Writing Code

### Before Coding
1. [ ] Understand existing code patterns
2. [ ] Check available libraries (don't reinvent)
3. [ ] Plan the structure
4. [ ] Consider error cases
5. [ ] Think about testing approach

### Coding Standards
1. [ ] Follow project conventions
2. [ ] Use descriptive variable names
3. [ ] Keep functions focused (one task)
4. [ ] Add docstrings to functions
5. [ ] Handle errors gracefully

### Import Management
```python
# Standard library imports first
import os
import sys
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from local_module import function
```

## Testing

### Before Running
1. [ ] Save all files
2. [ ] Check syntax: `python -m py_compile file.py`
3. [ ] Review imports
4. [ ] Verify file paths are correct
5. [ ] Check environment variables

### Testing Procedure
1. [ ] Test individual functions first
2. [ ] Test integration between components
3. [ ] Test edge cases
4. [ ] Test error handling
5. [ ] Test with realistic data

### Debugging Checklist
1. [ ] Read the full error message
2. [ ] Check the line number
3. [ ] Verify variable types
4. [ ] Print intermediate values
5. [ ] Use debugger if needed

## Code Review (Self)

### Before Committing
1. [ ] Remove debug print statements
2. [ ] Remove commented-out code
3. [ ] Check for hardcoded values
4. [ ] Verify no sensitive data
5. [ ] Run linting if available

### Code Quality
1. [ ] Is the code readable?
2. [ ] Are functions too long? (>50 lines)
3. [ ] Is there duplicated code?
4. [ ] Are errors handled?
5. [ ] Will someone else understand this?

## Platform-Specific Considerations

### Cross-Platform Compatibility
1. [ ] Use `os.path.join()` for file paths
2. [ ] Avoid platform-specific commands
3. [ ] Test on target platforms
4. [ ] Document platform requirements
5. [ ] Handle line endings correctly

### Windows Compatibility
1. [ ] No colons in filenames
2. [ ] No reserved characters (< > : " | ? *)
3. [ ] Case-insensitive filesystem aware
4. [ ] Path length limitations (260 chars)
5. [ ] Different path separators

## Performance Considerations

### Optimization Checklist
1. [ ] Profile before optimizing
2. [ ] Optimize algorithms first
3. [ ] Use appropriate data structures
4. [ ] Consider memory usage
5. [ ] Document performance requirements

### GPU/CUDA Development
1. [ ] Check CUDA availability first
2. [ ] Provide CPU fallback
3. [ ] Monitor GPU memory
4. [ ] Handle out-of-memory errors
5. [ ] Profile kernel performance

## Documentation

### Code Documentation
1. [ ] Module-level docstring
2. [ ] Function/class docstrings
3. [ ] Complex logic comments
4. [ ] TODO comments with context
5. [ ] Update documentation with code

### Project Documentation
1. [ ] README with quick start
2. [ ] Installation instructions
3. [ ] Usage examples
4. [ ] API documentation
5. [ ] Troubleshooting guide

## Security Checklist

### Never Do
- ❌ Hardcode credentials
- ❌ Log sensitive information
- ❌ Trust user input without validation
- ❌ Use eval() or exec() with user data
- ❌ Store passwords in plain text

### Always Do
- ✅ Use environment variables for secrets
- ✅ Validate all inputs
- ✅ Sanitize file paths
- ✅ Use secure random for security
- ✅ Keep dependencies updated

## Integration

### When Adding to Existing Project
1. [ ] Understand current architecture
2. [ ] Follow existing patterns
3. [ ] Check for existing utilities
4. [ ] Maintain backward compatibility
5. [ ] Update relevant documentation

### Library Selection
1. [ ] Check if already used in project
2. [ ] Verify license compatibility
3. [ ] Check maintenance status
4. [ ] Evaluate security record
5. [ ] Consider size/dependencies

## Deployment Preparation

### Pre-Deployment
1. [ ] Remove development-only code
2. [ ] Set production configurations
3. [ ] Test in production-like environment
4. [ ] Check resource requirements
5. [ ] Prepare rollback plan

### Edge Deployment (Jetson, etc.)
1. [ ] Check memory constraints
2. [ ] Optimize for target hardware
3. [ ] Test power consumption
4. [ ] Verify thermal limits
5. [ ] Plan for limited storage

## Common Patterns

### Experiment Structure
```python
class ExperimentName:
    def __init__(self):
        self.setup_logging()
        self.load_config()
        
    def run(self):
        try:
            results = self.execute()
            self.save_results(results)
            return results
        except Exception as e:
            self.log_error(e)
            raise
```

### Configuration Management
```python
# Use JSON/YAML for configuration
config = {
    "experiment": {
        "name": "test",
        "parameters": {...}
    }
}
```

### Result Saving
```python
# Always include metadata
results = {
    "timestamp": datetime.now().isoformat(),
    "version": "1.0",
    "parameters": config,
    "results": data,
    "environment": platform.platform()
}
```

## Post-Development

### Cleanup
1. [ ] Remove temporary files
2. [ ] Clear debug outputs
3. [ ] Update requirements.txt
4. [ ] Tag version if releasing
5. [ ] Archive old experiments

### Knowledge Transfer
1. [ ] Document lessons learned
2. [ ] Update team procedures
3. [ ] Share useful patterns
4. [ ] Note pitfalls discovered
5. [ ] Improve checklists

---

**Remember**: Good code is written once but read many times. Optimize for readability and maintainability over cleverness.