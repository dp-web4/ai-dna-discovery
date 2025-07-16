#!/usr/bin/env python3
"""
Continue Phase 3 - Run remaining experiments
"""

import time
import json
from datetime import datetime
from pathlib import Path

from experiment_tracker import PhaseManager
from phase_3_model_orchestra import (
    ConsensusBuildingExperiment,
    SpecializationDynamicsExperiment
)

def continue_phase_3():
    """Run remaining Phase 3 experiments"""
    print("=" * 60)
    print("PHASE 3: MODEL ORCHESTRA (Continuation)")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    print("\nContinuing with remaining experiments...\n")
    
    phase_manager = PhaseManager()
    
    # Run remaining experiments
    experiments = [
        ("Consensus Building", ConsensusBuildingExperiment),
        ("Specialization Dynamics", SpecializationDynamicsExperiment)
    ]
    
    for exp_name, ExpClass in experiments:
        print(f"\n{'='*50}")
        print(f"Starting: {exp_name}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        try:
            exp = ExpClass()
            exp.run()
            phase_manager.add_experiment(3, exp.tracker.get_summary())
            print(f"✓ {exp_name} completed successfully")
            
            # Brief pause between experiments
            time.sleep(5)
            
        except Exception as e:
            print(f"✗ {exp_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Mark phase complete
    phase_manager.complete_phase(3)
    
    # Generate phase report
    from phase_3_runner import generate_phase_3_report
    generate_phase_3_report()
    
    print("\n" + "="*60)
    print("PHASE 3 COMPLETED")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

if __name__ == "__main__":
    continue_phase_3()