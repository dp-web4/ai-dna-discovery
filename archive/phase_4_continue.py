#!/usr/bin/env python3
"""
Continue Phase 4 - Run remaining experiments
"""

import time
import json
from datetime import datetime
from pathlib import Path

from experiment_tracker import PhaseManager
from phase_4_energy_patterns import (
    InterferenceMappingExperiment,
    ConceptualCircuitsExperiment
)

def continue_phase_4():
    """Run remaining Phase 4 experiments"""
    print("=" * 60)
    print("PHASE 4: ENERGY/PATTERN DYNAMICS (Continuation)")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    print("\nContinuing with remaining experiments...\n")
    
    phase_manager = PhaseManager()
    
    # Check if interference mapping has a complete result file
    result_files = list(Path("phase_4_results").glob("interference_mapping_*.json"))
    completed_interference = any("checkpoint" not in str(f) for f in result_files)
    
    # Run remaining experiments
    experiments = []
    if not completed_interference:
        experiments.append(("Interference Mapping", InterferenceMappingExperiment))
    experiments.append(("Conceptual Circuits", ConceptualCircuitsExperiment))
    
    for exp_name, ExpClass in experiments:
        print(f"\n{'='*50}")
        print(f"Starting: {exp_name}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        try:
            exp = ExpClass()
            exp.run()
            phase_manager.add_experiment(4, exp.tracker.get_summary())
            print(f"✓ {exp_name} completed successfully")
            
            # Brief pause between experiments
            time.sleep(5)
            
        except Exception as e:
            print(f"✗ {exp_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Mark phase complete
    phase_manager.complete_phase(4)
    
    # Generate phase report
    from phase_4_runner import generate_phase_4_report
    generate_phase_4_report()
    
    print("\n" + "="*60)
    print("PHASE 4 COMPLETED")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

if __name__ == "__main__":
    continue_phase_4()