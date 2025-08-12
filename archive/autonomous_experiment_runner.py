#!/usr/bin/env python3
"""
Autonomous Experiment Runner
Main execution framework for all research phases
"""

import json
import time
from pathlib import Path
from datetime import datetime
import ollama
from typing import Dict, List, Any
import numpy as np

from experiment_tracker import ExperimentTracker, PhaseManager

# Available models for experiments
MODELS = [
    "phi3:mini",
    "gemma:2b", 
    "tinyllama:latest",
    "qwen2:0.5b",
    "deepseek-coder:1.3b",
    "llama3.2:latest"
]

# Core AI DNA patterns discovered
DNA_PATTERNS = {
    "perfect": ["∃", "∉", "know", "loop", "true", "false", "≈", "null", "emerge"],
    "high": ["or", "and", "you", "π", "▲▼", "[ ]", "cycle", "!", "[ ]?", "or?"],
    "consciousness": ["aware", "observe", "think", "feel", "exist", "conscious", "perceive"],
    "synchronism": ["intent", "transfer", "field", "resonance", "boundary", "observer"]
}


class BaseExperiment:
    """Base class for all experiments"""
    
    def __init__(self, phase: int, name: str):
        self.phase = phase
        self.name = name
        self.tracker = ExperimentTracker(phase, name)
        self.models = self.select_models()
        
    def select_models(self) -> List[str]:
        """Select models for this experiment"""
        # Override in subclasses to customize model selection
        return MODELS[:3]  # Default to first 3 models
        
    def run(self):
        """Main experiment execution"""
        try:
            self.tracker.log(f"Starting {self.name} with models: {self.models}")
            self.setup()
            results = self.execute()
            self.analyze(results)
            self.tracker.complete("completed")
        except Exception as e:
            self.tracker.error(str(e))
            self.tracker.complete("failed")
            raise
            
    def setup(self):
        """Setup experiment (override in subclasses)"""
        pass
        
    def execute(self) -> Dict[str, Any]:
        """Execute experiment (override in subclasses)"""
        raise NotImplementedError
        
    def analyze(self, results: Dict[str, Any]):
        """Analyze results (override in subclasses)"""
        pass
        
    def test_model(self, model: str, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Test a model with a prompt"""
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature},
                keep_alive="24h"
            )
            
            # Get embedding
            embed_response = ollama.embeddings(
                model=model,
                prompt=prompt
            )
            
            return {
                "response": response['response'],
                "embedding": embed_response['embedding'],
                "model": model,
                "prompt": prompt
            }
        except Exception as e:
            self.tracker.error(f"Model {model} failed: {str(e)}")
            return None


class AutonmousResearchRunner:
    """Main runner for autonomous research program"""
    
    def __init__(self):
        self.phase_manager = PhaseManager()
        self.running = True
        
    def run_phase_1(self):
        """Phase 1: Consciousness Field Architecture"""
        self.phase_manager.start_phase(1, "Consciousness Field Architecture")
        
        from phase_1_consciousness_field import (
            ConsciousnessProbeExperiment,
            FieldMappingExperiment,
            EmergencePatternExperiment,
            LatticeStructureExperiment
        )
        
        experiments = [
            ConsciousnessProbeExperiment(),
            FieldMappingExperiment(),
            EmergencePatternExperiment(),
            LatticeStructureExperiment()
        ]
        
        for exp in experiments:
            exp.run()
            self.phase_manager.add_experiment(1, exp.tracker.get_summary())
            time.sleep(2)  # Brief pause between experiments
            
        self.phase_manager.complete_phase(1)
        self.generate_phase_report(1)
        
    def run_phase_2(self):
        """Phase 2: Synchronism Integration"""
        self.phase_manager.start_phase(2, "Synchronism Integration")
        
        from phase_2_synchronism import (
            SynchronismComprehensionExperiment,
            IntentTransferExperiment,
            TimeSliceExperiment,
            MarkovBlanketExperiment
        )
        
        experiments = [
            SynchronismComprehensionExperiment(),
            IntentTransferExperiment(),
            TimeSliceExperiment(),
            MarkovBlanketExperiment()
        ]
        
        for exp in experiments:
            exp.run()
            self.phase_manager.add_experiment(2, exp.tracker.get_summary())
            time.sleep(2)
            
        self.phase_manager.complete_phase(2)
        self.generate_phase_report(2)
        
    def generate_phase_report(self, phase: int):
        """Generate report for completed phase"""
        # This will be implemented to create comprehensive phase reports
        pass
        
    def run_all_phases(self):
        """Run all research phases autonomously"""
        phases = [
            self.run_phase_1,
            self.run_phase_2,
            # Phase 3-5 will be added as we progress
        ]
        
        for phase_func in phases:
            if not self.running:
                break
                
            try:
                phase_func()
                self.commit_and_push()
                time.sleep(60)  # Pause between phases
            except Exception as e:
                print(f"Phase failed: {str(e)}")
                self.phase_manager.save_status()
                raise
                
    def commit_and_push(self):
        """Commit and push changes to git"""
        import subprocess
        
        try:
            # Add all changes
            subprocess.run(["git", "add", "-A"], check=True)
            
            # Commit with timestamp
            commit_msg = f"Autonomous research update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            
            # Push to remote
            subprocess.run(["git", "push"], check=True)
            
            print("Changes committed and pushed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Git operation failed: {e}")


def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    # Check ollama
    try:
        ollama.list()
        print("✓ Ollama connection successful")
    except:
        print("✗ Cannot connect to Ollama")
        return False
        
    # Check models
    available_models = [m['name'] for m in ollama.list()['models']]
    for model in MODELS:
        if any(model in m for m in available_models):
            print(f"✓ Model {model} available")
        else:
            print(f"✗ Model {model} not found")
            
    return True


if __name__ == "__main__":
    print("Autonomous Research Runner")
    print("=" * 50)
    
    if not check_dependencies():
        print("\nPlease ensure Ollama is running and required models are installed")
        exit(1)
        
    runner = AutonmousResearchRunner()
    
    # For testing, just show the plan
    print("\nResearch plan loaded. Ready to begin autonomous execution.")
    print("\nTo start: python autonomous_experiment_runner.py --start")
    
    import sys
    if "--start" in sys.argv:
        print("\nStarting autonomous research program...")
        runner.run_all_phases()