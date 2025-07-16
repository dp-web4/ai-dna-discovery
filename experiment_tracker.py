#!/usr/bin/env python3
"""
Autonomous Experiment Tracking System
Provides consistent logging, state management, and reporting across all phases
"""

import json
import os
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Any, Optional

class ExperimentTracker:
    def __init__(self, phase: int, experiment_name: str):
        self.phase = phase
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.results = {
            "phase": phase,
            "experiment": experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": None,
            "status": "running",
            "checkpoints": [],
            "results": {},
            "errors": [],
            "notes": []
        }
        
        # Create phase directory
        self.phase_dir = Path(f"phase_{phase}_results")
        self.phase_dir.mkdir(exist_ok=True)
        
        # Set up file paths
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.results_file = self.phase_dir / f"{experiment_name}_{timestamp}.json"
        self.checkpoint_file = self.phase_dir / f"{experiment_name}_checkpoint.json"
        
        # Initialize experiment log
        self.log(f"Started experiment: {experiment_name}")
        
    def log(self, message: str, level: str = "INFO"):
        """Add timestamped log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.results["notes"].append(entry)
        print(f"[{level}] {message}")
        
    def checkpoint(self, data: Dict[str, Any], name: str = ""):
        """Save experiment checkpoint"""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "name": name or f"checkpoint_{len(self.results['checkpoints'])}",
            "data": data
        }
        self.results["checkpoints"].append(checkpoint)
        
        # Save checkpoint file
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"Checkpoint saved: {checkpoint['name']}")
        
    def record_result(self, key: str, value: Any):
        """Record a result value"""
        self.results["results"][key] = value
        
    def error(self, error_msg: str):
        """Record an error"""
        self.results["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": error_msg
        })
        self.log(f"ERROR: {error_msg}", "ERROR")
        
    def complete(self, status: str = "completed"):
        """Mark experiment as complete and save final results"""
        self.results["end_time"] = datetime.now().isoformat()
        self.results["status"] = status
        
        # Calculate duration
        duration = (datetime.now() - self.start_time).total_seconds()
        self.results["duration_seconds"] = duration
        
        # Save final results
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        self.log(f"Experiment completed with status: {status}")
        self.log(f"Results saved to: {self.results_file}")
        
        # Clean up checkpoint file
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        return {
            "phase": self.phase,
            "experiment": self.experiment_name,
            "status": self.results["status"],
            "start_time": self.results["start_time"],
            "end_time": self.results["end_time"],
            "duration_seconds": self.results.get("duration_seconds", 0),
            "num_checkpoints": len(self.results["checkpoints"]),
            "num_errors": len(self.results["errors"]),
            "results_summary": list(self.results["results"].keys())
        }


class PhaseManager:
    """Manages overall phase execution and reporting"""
    
    def __init__(self):
        self.status_file = Path("autonomous_experiment_status.json")
        self.load_status()
        
    def load_status(self):
        """Load current experiment status"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                self.status = json.load(f)
        else:
            self.status = {
                "current_phase": 1,
                "phase_status": {},
                "start_date": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
            
    def save_status(self):
        """Save current status"""
        self.status["last_update"] = datetime.now().isoformat()
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
            
    def start_phase(self, phase: int, description: str):
        """Mark a phase as started"""
        self.status["current_phase"] = phase
        self.status["phase_status"][str(phase)] = {
            "description": description,
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
            "experiments": []
        }
        self.save_status()
        
    def complete_phase(self, phase: int):
        """Mark a phase as complete"""
        if str(phase) in self.status["phase_status"]:
            self.status["phase_status"][str(phase)]["status"] = "completed"
            self.status["phase_status"][str(phase)]["end_time"] = datetime.now().isoformat()
        self.save_status()
        
    def add_experiment(self, phase: int, experiment_summary: Dict[str, Any]):
        """Add experiment summary to phase"""
        if str(phase) in self.status["phase_status"]:
            self.status["phase_status"][str(phase)]["experiments"].append(experiment_summary)
        self.save_status()
        
    def get_progress_report(self) -> str:
        """Generate progress report"""
        report = ["# Autonomous Experiment Progress Report\n"]
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Current Phase:** {self.status['current_phase']}\n")
        
        for phase_num, phase_data in sorted(self.status["phase_status"].items()):
            report.append(f"\n## Phase {phase_num}: {phase_data['description']}")
            report.append(f"**Status:** {phase_data['status']}")
            report.append(f"**Started:** {phase_data['start_time']}")
            
            if 'end_time' in phase_data:
                report.append(f"**Completed:** {phase_data['end_time']}")
                
            if phase_data['experiments']:
                report.append("\n### Experiments:")
                for exp in phase_data['experiments']:
                    report.append(f"- **{exp['experiment']}**: {exp['status']} "
                                f"({exp.get('duration_seconds', 0):.1f}s)")
                    
        return "\n".join(report)


def test_tracker():
    """Test the tracking system"""
    print("Testing experiment tracker...")
    
    # Create test experiment
    tracker = ExperimentTracker(0, "test_experiment")
    
    # Simulate some work
    tracker.log("Starting test sequence")
    time.sleep(1)
    
    # Record some results
    tracker.record_result("test_value", 42)
    tracker.record_result("test_list", [1, 2, 3])
    
    # Create checkpoint
    tracker.checkpoint({"iteration": 1, "score": 0.5})
    
    # Complete
    tracker.complete()
    
    # Test phase manager
    pm = PhaseManager()
    pm.start_phase(0, "Test Phase")
    pm.add_experiment(0, tracker.get_summary())
    pm.complete_phase(0)
    
    print("\nProgress Report:")
    print(pm.get_progress_report())
    
    print("\nTracker test completed successfully!")


if __name__ == "__main__":
    test_tracker()