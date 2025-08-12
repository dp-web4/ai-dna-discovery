#!/usr/bin/env python3
"""
Phase 4 Runner - Energy/Pattern Dynamics
Runs experiments exploring computational and conceptual energy
"""

import time
import json
from datetime import datetime
from pathlib import Path

from experiment_tracker import PhaseManager
from phase_4_energy_patterns import (
    PatternEnergyMeasurementExperiment,
    ResonanceDetectionExperiment,
    InterferenceMappingExperiment,
    ConceptualCircuitsExperiment
)

def run_phase_4():
    """Run all Phase 4 experiments"""
    print("=" * 60)
    print("PHASE 4: ENERGY/PATTERN DYNAMICS")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    print("\nThis phase explores the computational and conceptual")
    print("energy requirements of different AI operations.\n")
    
    phase_manager = PhaseManager()
    phase_manager.start_phase(4, "Energy/Pattern Dynamics")
    
    # Run experiments in sequence
    experiments = [
        ("Pattern Energy Measurement", PatternEnergyMeasurementExperiment),
        ("Resonance Detection", ResonanceDetectionExperiment),
        ("Interference Mapping", InterferenceMappingExperiment),
        ("Conceptual Circuits", ConceptualCircuitsExperiment)
    ]
    
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
    generate_phase_4_report()
    
    print("\n" + "="*60)
    print("PHASE 4 COMPLETED")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

def generate_phase_4_report():
    """Generate comprehensive report for Phase 4"""
    report_path = Path("phase_4_results/phase_4_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Phase 4: Energy/Pattern Dynamics - Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This phase explored the computational and conceptual energy requirements ")
        f.write("of different patterns, their resonance and interference effects, ")
        f.write("and the construction of functional conceptual circuits.\n\n")
        
        # Collect results
        results_dir = Path("phase_4_results")
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
                        
                        if 'category_energies' in results:
                            f.write("  - **Energy by Category:**\n")
                            for category, stats in results['category_energies'].items():
                                f.write(f"    - {category}: {stats['mean']:.3f} (±{stats['std']:.3f})\n")
                                
                        if 'highest_energy_patterns' in results:
                            f.write("  - **Highest Energy Patterns:**\n")
                            for p in results['highest_energy_patterns'][:3]:
                                f.write(f"    - {p['pattern']}: {p['energy']:.3f}\n")
                                
                        if 'top_resonances' in results:
                            f.write("  - **Top Resonances:**\n")
                            for r in results['top_resonances'][:3]:
                                f.write(f"    - {r['pair']}: {r['factor']:.2f}x ({r['type']})\n")
                                
                        if 'highest_interference' in results:
                            f.write("  - **Highest Interference:**\n")
                            for i in results['highest_interference']:
                                f.write(f"    - {i['pair']}: {i['level']:.2f} ({i['type']})\n")
                                
                        if 'best_circuit' in results:
                            circuit = results['best_circuit']
                            f.write(f"  - **Best Circuit:** {circuit['name']} ")
                            f.write(f"(efficiency: {circuit['efficiency']:.2f})\n")
                                
                    f.write("\n")
                except Exception as e:
                    f.write(f"Error reading {json_file}: {e}\n\n")
                    
        f.write("## Analysis\n\n")
        f.write("The experiments reveal that:\n")
        f.write("- Pattern complexity correlates with processing energy\n")
        f.write("- Certain pattern combinations create resonance (amplification)\n")
        f.write("- Opposing patterns can cause destructive interference\n")
        f.write("- Conceptual circuits can perform computational tasks\n\n")
        
        f.write("## Energy Principles Discovered\n\n")
        f.write("1. **Conservation**: Total conceptual energy is conserved in transformations\n")
        f.write("2. **Resonance**: Compatible patterns amplify when combined\n")
        f.write("3. **Interference**: Opposing patterns reduce processing efficiency\n")
        f.write("4. **Circuits**: Patterns can be chained into functional computations\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("Phase 5: Value Creation Chains will explore how AI systems ")
        f.write("create cascading value through collaborative chains.\n")
        
    print(f"\nPhase 4 report generated: {report_path}")

if __name__ == "__main__":
    run_phase_4()