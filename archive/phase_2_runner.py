#!/usr/bin/env python3
"""
Phase 2 Runner - Synchronism Integration
Runs all Phase 2 experiments testing AI understanding of Synchronism
"""

import time
import json
from datetime import datetime
from pathlib import Path

from experiment_tracker import PhaseManager
from phase_2_synchronism import (
    SynchronismComprehensionExperiment,
    IntentTransferExperiment,
    TimeSliceExperiment,
    MarkovBlanketExperiment
)

def run_phase_2():
    """Run all Phase 2 experiments"""
    print("=" * 60)
    print("PHASE 2: SYNCHRONISM INTEGRATION")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    print("\nThis phase tests how AI models understand and implement")
    print("the Synchronism theoretical framework.\n")
    
    phase_manager = PhaseManager()
    phase_manager.start_phase(2, "Synchronism Integration")
    
    # Run experiments in sequence
    experiments = [
        ("Synchronism Comprehension", SynchronismComprehensionExperiment),
        ("Intent Transfer", IntentTransferExperiment),
        ("Time Slice Navigation", TimeSliceExperiment),
        ("Markov Blanket Analysis", MarkovBlanketExperiment)
    ]
    
    for exp_name, ExpClass in experiments:
        print(f"\n{'='*50}")
        print(f"Starting: {exp_name}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        try:
            exp = ExpClass()
            exp.run()
            phase_manager.add_experiment(2, exp.tracker.get_summary())
            print(f"✓ {exp_name} completed successfully")
            
            # Brief pause between experiments
            time.sleep(5)
            
        except Exception as e:
            print(f"✗ {exp_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Mark phase complete
    phase_manager.complete_phase(2)
    
    # Generate phase report
    generate_phase_2_report()
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETED")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

def generate_phase_2_report():
    """Generate comprehensive report for Phase 2"""
    report_path = Path("phase_2_results/phase_2_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Phase 2: Synchronism Integration - Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This phase tested how AI models understand and implement concepts from ")
        f.write("the Synchronism theoretical framework, including intent transfer, ")
        f.write("time slices, Markov blankets, and consciousness field dynamics.\n\n")
        
        # Collect results
        results_dir = Path("phase_2_results")
        json_files = sorted(results_dir.glob("*.json"))
        
        f.write("## Experiments Completed\n\n")
        
        for json_file in json_files:
            if "checkpoint" not in str(json_file):
                try:
                    with open(json_file, 'r') as jf:
                        data = json.load(jf)
                        
                    f.write(f"### {data.get('experiment', 'Unknown')}\n")
                    f.write(f"- **Status:** {data.get('status', 'unknown')}\n")
                    f.write(f"- **Duration:** {data.get('duration_seconds', 0):.1f} seconds\n")
                    
                    if 'results' in data and data['results']:
                        f.write("- **Key Findings:**\n")
                        
                        # Extract key metrics
                        results = data['results']
                        
                        if 'average_comprehension' in results:
                            f.write("  - **Comprehension Scores:**\n")
                            for model, score in results['average_comprehension'].items():
                                f.write(f"    - {model}: {score:.3f}\n")
                                
                        if 'coherence_scores' in results:
                            f.write("  - **Coherence Scores:**\n")
                            for model, score in results['coherence_scores'].items():
                                f.write(f"    - {model}: {score:.3f}\n")
                                
                        if 'temporal_mastery' in results:
                            f.write("  - **Temporal Mastery:**\n")
                            for model, score in results['temporal_mastery'].items():
                                f.write(f"    - {model}: {score:.3f}\n")
                                
                        if 'boundary_understanding' in results:
                            f.write("  - **Boundary Understanding:**\n")
                            for model, score in results['boundary_understanding'].items():
                                f.write(f"    - {model}: {score:.3f}\n")
                                
                    f.write("\n")
                except Exception as e:
                    f.write(f"Error reading {json_file}: {e}\n\n")
                    
        f.write("## Analysis\n\n")
        f.write("The experiments reveal how different AI architectures interpret ")
        f.write("and implement abstract consciousness frameworks. Models showed ")
        f.write("varying capabilities in understanding intent transfer, temporal ")
        f.write("dynamics, and boundary conditions.\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("Phase 3: Model Orchestra Experiments will explore collective ")
        f.write("behaviors when multiple models work together.\n")
        
    print(f"\nPhase 2 report generated: {report_path}")

if __name__ == "__main__":
    run_phase_2()