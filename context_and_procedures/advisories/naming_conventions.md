# Naming Conventions Advisory

*Last Updated: 2025-01-24*

## Core Principles

1. **Clarity**: Names should clearly indicate purpose
2. **Consistency**: Same patterns throughout project
3. **Cross-platform**: Works on Windows, Linux, Mac
4. **Searchable**: Easy to find with grep/search
5. **Sortable**: Logical ordering when listed

## File Naming

### General Rules
- Use lowercase
- Separate words with underscores
- No spaces in filenames
- No special characters (especially : < > | ? * " ')
- Keep under 255 characters
- Use ISO dates (YYYY-MM-DD or YYYYMMDD)

### File Type Conventions

#### Python Files
```
module_name.py              # Regular modules
test_module_name.py         # Test files
__init__.py                 # Package markers
config.py                   # Configuration
utils.py                    # Utilities
```

#### Data Files
```
dataset_YYYYMMDD.csv        # Dated data
experiment_YYYYMMDD_HHMMSS.json  # Timestamped results
model_checkpoint_epoch_N.pth     # Versioned checkpoints
processed_data_v2.parquet       # Version numbered
```

#### Documentation
```
README.md                   # Standard readme
CONTRIBUTING.md             # Contribution guide
API_REFERENCE.md           # API documentation
phase_1_report.md          # Numbered phases
meeting_notes_2025-01-24.md # Dated notes
```

#### Configuration
```
config.yaml                 # Main config
config_dev.yaml            # Environment specific
.env.example               # Example environment
settings.json              # Application settings
```

## Directory Naming

### Standard Directories
```
src/          # Source code
tests/        # Test files
docs/         # Documentation  
data/         # Data files
results/      # Experiment results
scripts/      # Utility scripts
config/       # Configuration files
assets/       # Images, etc.
```

### No Plurals vs Plurals
- Use plurals for collections: `tests/`, `docs/`, `results/`
- Use singular for single purpose: `config/`, `data/`
- Be consistent within project

## Variable Naming (Python)

### Conventions by Type
```python
# Variables: lowercase with underscores
user_name = "John"
is_valid = True
total_count = 42

# Constants: uppercase with underscores
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
API_KEY = os.getenv("API_KEY")

# Functions: lowercase with underscores
def calculate_score():
    pass

def get_user_by_id(user_id):
    pass

# Classes: PascalCase
class UserModel:
    pass

class DataProcessor:
    pass

# Private: prefix with underscore
_internal_state = {}
def _helper_function():
    pass
```

### Descriptive Names
```python
# Bad
def calc(x, y):
    return x * y * 0.1

# Good  
def calculate_discount_amount(price, quantity):
    DISCOUNT_RATE = 0.1
    return price * quantity * DISCOUNT_RATE
```

## Common Patterns

### Timestamps
```python
# ISO 8601 format
timestamp = datetime.now().isoformat()  # 2025-01-24T10:30:45.123456

# For filenames (no colons!)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 20250124_103045
```

### Versioning
```
# Semantic versioning
v1.0.0
v2.1.3

# For files
model_v1.pth
config_v2.yaml
dataset_v3_final.csv  # Avoid "final" - it's never final!
```

### Experiments
```
# Include what makes it unique
experiment_bert_lr0.001_epoch50.json
results_baseline_2025-01-24.csv
checkpoint_transformer_step_1000.pth
```

## Model and AI Specific

### Model Names
```python
# Bad - uses colon
model_name = "phi3:mini"  # Breaks on Windows!

# Good - use underscore
model_name = "phi3_mini"
model_name = "phi3-mini"  # Also OK
```

### Result Files
```
# Include key parameters
results_phi3_mini_temp0.7_2025-01-24.json
embeddings_bert_layer12_frozen.npz
predictions_ensemble_voting_test.csv
```

### Checkpoints
```
checkpoint_epoch_10.pth
checkpoint_step_5000.pth  
checkpoint_best_val_loss.pth
checkpoint_final.pth
```

## Cross-Platform Considerations

### Path Construction
```python
# Bad - hardcoded paths
path = "data/raw/file.csv"  # Breaks on Windows
path = "data\\raw\\file.csv"  # Breaks on Unix

# Good - use os.path or pathlib
import os
path = os.path.join("data", "raw", "file.csv")

from pathlib import Path
path = Path("data") / "raw" / "file.csv"
```

### Reserved Names
Avoid these on ALL platforms:
- Windows: CON, PRN, AUX, NUL, COM1-9, LPT1-9
- Unix: Files starting with . are hidden
- General: Don't end with . or space

## Git Branch Naming

### Branch Patterns
```
feature/add-user-auth
bugfix/fix-memory-leak
hotfix/security-patch
experiment/new-architecture
release/v2.0.0
```

### Include Issue Numbers
```
feature/123-add-payment-processing
bugfix/456-fix-login-error
```

## Common Mistakes to Avoid

### Problematic Names
```
# Too vague
test.py
data.csv
results.json
utils.py  # OK if truly general utilities

# Too long
this_is_the_final_version_of_the_model_after_extensive_testing_v2_final_final.pth

# Platform issues  
model:mini.pth      # Colon breaks Windows
my model.pth        # Spaces cause issues
model?.pth          # Special chars problematic
```

### Timestamp Mistakes
```python
# Bad - includes colons
timestamp = str(datetime.now())  # 2025-01-24 10:30:45.123456

# Good - filesystem safe
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

## Best Practices Summary

1. **Be Consistent**: Pick a style and stick to it
2. **Be Descriptive**: `user_auth_handler.py` > `handler.py`
3. **Be Platform Aware**: Test on target platforms
4. **Be Future Proof**: Consider how names will sort/search
5. **Be Collaborative**: Document your conventions

## Quick Reference

### Allowed Characters
- Letters (a-z, A-Z)
- Numbers (0-9)  
- Underscore (_)
- Hyphen (-)
- Period (.)

### Forbidden Characters
- Colon (:)
- Less than (<)
- Greater than (>)
- Quote (" ')
- Pipe (|)
- Question mark (?)
- Asterisk (*)
- Forward slash (/)
- Backslash (\)

---

**Remember**: Good naming is a gift to your future self and your collaborators. The extra seconds spent choosing a good name save hours of confusion later.