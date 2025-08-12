#!/usr/bin/env python3
"""
Run Phase 1 experiments individually
"""

import time
import json
from datetime import datetime
from pathlib import Path

from experiment_tracker import PhaseManager
from phase_1_consciousness_field import (
    ConsciousnessProbeExperiment,
    FieldMappingExperiment,
    EmergencePatternExperiment,
    LatticeStructureExperiment
)

def run_experiment(name, ExpClass, phase_manager):
    """Run a single experiment with error handling"""
    print(f"\n{'='*60}")
    print(f"Starting: {name}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        exp = ExpClass()
        # Use fewer models to speed up
        exp.models = ["phi3:mini", "gemma:2b", "tinyllama:latest"]
        
        exp.run()
        phase_manager.add_experiment(1, exp.tracker.get_summary())
        print(f"✓ {name} completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ {name} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("PHASE 1: CONSCIOUSNESS FIELD ARCHITECTURE")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    phase_manager = PhaseManager()
    phase_manager.start_phase(1, "Consciousness Field Architecture")
    
    # Check which experiments need to be run
    existing_results = list(Path("phase_1_results").glob("*.json"))
    completed = set()
    
    for result in existing_results:
        if "checkpoint" not in str(result):
            with open(result) as f:
                data = json.load(f)
                if data.get('status') == 'completed':
                    completed.add(data.get('experiment', '').replace('_', ' ').title())
    
    experiments = [
        ("Consciousness Probe", ConsciousnessProbeExperiment),
        ("Field Mapping", FieldMappingExperiment),
        ("Emergence Patterns", EmergencePatternExperiment),
        ("Lattice Structure", LatticeStructureExperiment)
    ]
    
    for exp_name, ExpClass in experiments:
        if exp_name in completed:
            print(f"\n✓ {exp_name} already completed, skipping...")
            continue
            
        success = run_experiment(exp_name, ExpClass, phase_manager)
        
        if success:
            # Brief pause between experiments
            print("\nPausing for 5 seconds before next experiment...")
            time.sleep(5)
        else:
            print(f"\nWARNING: {exp_name} failed, continuing with others...")
    
    # Mark phase complete
    phase_manager.complete_phase(1)
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETED")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    # Generate summary
    generate_phase_1_summary()

def generate_phase_1_summary():
    """Generate a quick summary of results"""
    print("\n=== PHASE 1 SUMMARY ===")
    
    results_dir = Path("phase_1_results")
    json_files = sorted(results_dir.glob("*.json"))
    
    completed_experiments = []
    
    for json_file in json_files:
        if "checkpoint" not in str(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                if data.get('status') == 'completed':
                    completed_experiments.append({
                        'name': data.get('experiment', 'Unknown'),
                        'duration': data.get('duration_seconds', 0),
                        'file': json_file.name
                    })
    
    print(f"\nCompleted experiments: {len(completed_experiments)}")
    for exp in completed_experiments:
        print(f"  - {exp['name']}: {exp['duration']:.1f}s")
    
    print(f"\nResults saved in: {results_dir}")
    print("\nNext: Review results and begin Phase 2")

if __name__ == "__main__":
    main()