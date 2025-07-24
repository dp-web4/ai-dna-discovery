# Project Structure Advisory

*Last Updated: 2025-01-24*

## Recommended Project Organization

### Basic Structure
```
project_name/
├── README.md              # Project overview and quick start
├── requirements.txt       # Python dependencies
├── .gitignore            # MUST be created first
├── src/                  # Source code
│   ├── __init__.py
│   └── modules/
├── tests/                # Test files
├── docs/                 # Documentation
├── data/                 # Data files (git-ignored if large)
├── results/              # Experiment results
├── scripts/              # Utility scripts
└── config/               # Configuration files
```

### For Research Projects
```
research_project/
├── README.md
├── AUTONOMOUS_RESEARCH_PLAN.md
├── requirements.txt
├── .gitignore
├── experiments/          # Experiment code
│   ├── base_experiment.py
│   ├── phase_1/
│   ├── phase_2/
│   └── ...
├── results/              # Organized by phase
│   ├── phase_1_results/
│   ├── phase_2_results/
│   └── synthesis/
├── visualizations/       # Generated plots
├── reports/              # Written documentation
└── CUMULATIVE_PROGRESS_REPORT.md
```

### For AI/ML Projects
```
ai_project/
├── README.md
├── requirements.txt
├── .gitignore
├── models/               # Model definitions
├── training/             # Training scripts
├── inference/            # Inference code
├── data/                 # Datasets (usually git-ignored)
│   ├── raw/
│   ├── processed/
│   └── README.md        # Explains data sources
├── weights/              # Trained models (git-ignored)
├── configs/              # Training/model configs
├── notebooks/            # Jupyter notebooks
└── utils/                # Helper functions
```

## Best Practices

### README.md Essentials
1. **Project Title and Description**
2. **Quick Start** (installation and basic usage)
3. **Requirements** (hardware, software, dependencies)
4. **Project Structure** explanation
5. **Usage Examples**
6. **Results Summary** (if applicable)
7. **Contributing Guidelines**
8. **License Information**

### Dependency Management
```bash
# Create requirements.txt
pip freeze > requirements.txt

# Better: only direct dependencies
pip install pipreqs
pipreqs . --force

# For different environments
requirements.txt          # Production
requirements-dev.txt      # Development only
requirements-test.txt     # Testing only
```

### Configuration Files
```python
# config/default.yaml
experiment:
  name: "baseline"
  seed: 42
  
model:
  type: "transformer"
  parameters:
    hidden_size: 768
    
paths:
  data: "./data"
  results: "./results"
```

### Data Organization
- **Raw Data**: Original, unmodified files
- **Processed Data**: Cleaned, formatted data
- **Metadata**: Data descriptions, sources
- **Splits**: train/val/test divisions
- **Documentation**: How data was collected/processed

## Version Control Strategy

### What to Track
✅ Source code
✅ Documentation
✅ Configuration files
✅ Small data samples
✅ Analysis notebooks
✅ Result summaries

### What NOT to Track
❌ Virtual environments
❌ Large datasets
❌ Model weights
❌ Generated outputs
❌ Temporary files
❌ Cache directories

### Branch Strategy
```
main/master     # Stable, tested code
develop         # Integration branch
feature/xyz     # New features
bugfix/abc      # Bug fixes
experiment/123  # Experimental work
```

## Documentation Standards

### Code Documentation
```python
"""
Module description goes here.
"""

def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input provided
    """
    pass
```

### Experiment Documentation
Each experiment should have:
1. **Purpose**: What question does it answer?
2. **Hypothesis**: What do we expect?
3. **Method**: How will we test it?
4. **Results**: What happened?
5. **Conclusions**: What did we learn?

## File Naming Conventions

### General Rules
- Use lowercase with underscores
- Be descriptive but concise
- Include dates in ISO format (YYYY-MM-DD)
- No spaces or special characters
- Add version numbers when relevant

### Examples
```
# Good
experiment_results_2025-01-24.json
phase_1_consciousness_detection.py
model_checkpoint_epoch_100.pth

# Bad
test.py
results.json
New Document.docx
model:final.pth  # Colon breaks Windows!
```

## Testing Structure

### Test Organization
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test data
└── test_*.py       # Test files (prefix with test_)
```

### Test Naming
```python
def test_function_name_condition_expected():
    """Test that function handles condition correctly."""
    pass

# Examples
def test_load_model_missing_file_raises_error():
    pass

def test_consciousness_score_valid_input_returns_float():
    pass
```

## Performance Considerations

### Large Files
- Store large files externally (S3, Google Drive)
- Use `.gitignore` to exclude them
- Provide download scripts
- Document file sources

### Results Storage
```python
# Efficient result storage
results = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
        "git_hash": get_git_hash()
    },
    "config": config_dict,
    "metrics": metrics_dict,
    "raw_data": "path/to/large/file.npz"  # Reference, not embed
}
```

## Security Considerations

### Secrets Management
```python
# .env file (git-ignored)
API_KEY=your_secret_key
DATABASE_URL=postgresql://user:pass@host/db

# Python code
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
```

### Sensitive Data
- Never commit real credentials
- Use example files (.env.example)
- Sanitize data before sharing
- Document security requirements

---

**Remember**: Good project structure makes collaboration easier, reduces errors, and speeds up development. Start with good habits from day one.