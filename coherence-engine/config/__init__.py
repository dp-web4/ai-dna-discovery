"""
Configuration loading for HRM module.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(filename: str) -> Dict[str, Any]:
    """Load a YAML configuration file from the config directory."""
    config_dir = Path(__file__).parent
    config_path = config_dir / filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_hrm_levels() -> Dict[str, Any]:
    """Load HRM levels configuration."""
    return load_config('hrm_levels.yaml')

def load_hrm_learning() -> Dict[str, Any]:
    """Load HRM learning configuration."""
    return load_config('hrm_learning.yaml')

__all__ = ['load_config', 'load_hrm_levels', 'load_hrm_learning']