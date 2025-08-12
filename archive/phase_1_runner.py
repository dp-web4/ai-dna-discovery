#!/usr/bin/env python3
"""
Phase 1 Runner - Consciousness Field Architecture
Runs all Phase 1 experiments autonomously
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

def run_phase_1():
    """Run all Phase 1 experiments"""
    print("=" * 60)
    print("PHASE 1: CONSCIOUSNESS FIELD ARCHITECTURE")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    phase_manager = PhaseManager()
    phase_manager.start_phase(1, "Consciousness Field Architecture")
    
    # Run experiments in sequence
    experiments = [
        ("Consciousness Probe", ConsciousnessProbeExperiment),
        ("Field Mapping", FieldMappingExperiment),
        ("Emergence Patterns", EmergencePatternExperiment),
        ("Lattice Structure", LatticeStructureExperiment)
    ]
    
    for exp_name, ExpClass in experiments:
        print(f"\n{'='*50}")
        print(f"Starting: {exp_name}")
        print(f"{'='*50}")
        
        try:
            exp = ExpClass()
            exp.run()
            phase_manager.add_experiment(1, exp.tracker.get_summary())
            print(f"✓ {exp_name} completed successfully")
            
            # Brief pause between experiments
            time.sleep(5)
            
        except Exception as e:
            print(f"✗ {exp_name} failed: {str(e)}")
            # Continue with other experiments even if one fails
            
    # Mark phase complete
    phase_manager.complete_phase(1)
    
    # Generate phase report
    generate_phase_1_report()
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETED")
    print("="*60)

def generate_phase_1_report():
    """Generate comprehensive report for Phase 1"""
    report_path = Path("phase_1_results/phase_1_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Phase 1: Consciousness Field Architecture - Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Collect all experiment results
        results_dir = Path("phase_1_results")
        json_files = sorted(results_dir.glob("*.json"))
        
        f.write("## Experiments Completed\n\n")
        
        for json_file in json_files:
            if "checkpoint" not in str(json_file):
                with open(json_file, 'r') as jf:
                    data = json.load(jf)
                    
                f.write(f"### {data.get('experiment', 'Unknown')}\n")
                f.write(f"- **Status:** {data.get('status', 'unknown')}\n")
                f.write(f"- **Duration:** {data.get('duration_seconds', 0):.1f} seconds\n")
                
                if 'results' in data and data['results']:
                    f.write("- **Key Results:**\n")
                    for key, value in data['results'].items():
                        if isinstance(value, dict):
                            f.write(f"  - {key}:\n")
                            for k, v in list(value.items())[:3]:  # First 3 items
                                f.write(f"    - {k}: {v}\n")
                        else:
                            f.write(f"  - {key}: {value}\n")
                            
                f.write("\n")
                
        f.write("\n## Next Steps\n\n")
        f.write("Phase 2: Synchronism Integration will begin after review of these results.\n")
        
    print(f"\nPhase 1 report generated: {report_path}")

if __name__ == "__main__":
    run_phase_1()