#!/usr/bin/env python3
"""
Run layer dynamics experiment with sequential model loading
Works within current Ollama limits
"""

import subprocess
import time
import sys

# Add venv to path
sys.path.append('venv/lib/python3.12/site-packages')

from layer_dynamics_experiment import LayerDynamicsExperiment

def run_with_model_management():
    """Run experiment with careful model management"""
    
    print("Starting Layer Dynamics Experiment (Sequential Mode)")
    print("="*60)
    
    experiment = LayerDynamicsExperiment()
    
    # Reduce models to fit in memory
    experiment.models = [
        "deepseek-coder:1.3b",  # Primary candidate (already loaded)
        "gemma:2b",             # Control (already loaded)
        "tinyllama:latest"      # Control (already loaded)
    ]
    
    print(f"Testing with {len(experiment.models)} models to fit GPU memory")
    print("Models:", ", ".join(experiment.models))
    
    # Run the experiment
    experiment.run_full_experiment()

if __name__ == "__main__":
    run_with_model_management()