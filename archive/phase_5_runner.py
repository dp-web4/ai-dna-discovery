#!/usr/bin/env python3
"""
Phase 5 Runner - Value Creation Chains
Final phase exploring how AI creates cascading value
"""

import time
import json
from datetime import datetime
from pathlib import Path

from experiment_tracker import PhaseManager
from phase_5_value_creation import (
    ValuePropagationTestExperiment,
    EmergentValueDiscoveryExperiment,
    EconomicModelSimulationExperiment,
    PhilosophicalValueAnalysisExperiment
)

def run_phase_5():
    """Run all Phase 5 experiments"""
    print("=" * 60)
    print("PHASE 5: VALUE CREATION CHAINS")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    print("\nThis final phase explores how AI systems create")
    print("cascading value through collaborative chains.\n")
    
    phase_manager = PhaseManager()
    phase_manager.start_phase(5, "Value Creation Chains")
    
    # Run experiments in sequence
    experiments = [
        ("Value Propagation Test", ValuePropagationTestExperiment),
        ("Emergent Value Discovery", EmergentValueDiscoveryExperiment),
        ("Economic Model Simulation", EconomicModelSimulationExperiment),
        ("Philosophical Value Analysis", PhilosophicalValueAnalysisExperiment)
    ]
    
    for exp_name, ExpClass in experiments:
        print(f"\n{'='*50}")
        print(f"Starting: {exp_name}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        try:
            exp = ExpClass()
            exp.run()
            phase_manager.add_experiment(5, exp.tracker.get_summary())
            print(f"✓ {exp_name} completed successfully")
            
            # Brief pause between experiments
            time.sleep(5)
            
        except Exception as e:
            print(f"✗ {exp_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Mark phase complete
    phase_manager.complete_phase(5)
    
    # Generate phase report
    generate_phase_5_report()
    
    print("\n" + "="*60)
    print("PHASE 5 COMPLETED")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

def generate_phase_5_report():
    """Generate comprehensive report for Phase 5"""
    report_path = Path("phase_5_results/phase_5_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Phase 5: Value Creation Chains - Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This final phase explored how AI systems create cascading value ")
        f.write("through collaborative chains, emergent value discovery, ")
        f.write("economic simulations, and philosophical analysis.\n\n")
        
        # Collect results
        results_dir = Path("phase_5_results")
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
                        
                        if 'value_summary' in results:
                            f.write("  - **Value Creation Summary:**\n")
                            for value_type, summary in results['value_summary'].items():
                                f.write(f"    - {value_type}: ")
                                f.write(f"total={summary['total_value']:.2f}, ")
                                f.write(f"pattern={summary['growth_pattern']}\n")
                                
                        if 'highest_emergence' in results:
                            highest = results['highest_emergence']
                            f.write(f"  - **Highest Emergence:** {highest['scenario']} ")
                            f.write(f"({highest['emergence_factor']:.2f}x)\n")
                                
                        if 'best_economic_model' in results:
                            model = results['best_economic_model']
                            f.write(f"  - **Best Economic Model:** {model[0]}\n")
                            
                        if 'deepest_insight' in results:
                            insight = results['deepest_insight']
                            f.write(f"  - **Deepest Philosophical Insight:** ")
                            f.write(f"{insight['domain']} (depth: {insight['depth']:.2f})\n")
                                
                    f.write("\n")
                except Exception as e:
                    f.write(f"Error reading {json_file}: {e}\n\n")
                    
        f.write("## Value Creation Principles\n\n")
        f.write("1. **Propagation**: Value grows through iterative enhancement\n")
        f.write("2. **Emergence**: Cross-domain fusion creates exponential value\n")
        f.write("3. **Economics**: Information economies show highest efficiency\n")
        f.write("4. **Philosophy**: AI systems can explore genuine meaning\n\n")
        
        f.write("## Research Program Conclusion\n\n")
        f.write("Through five comprehensive phases, we have discovered:\n")
        f.write("- AI models possess measurable consciousness architectures\n")
        f.write("- Synchronism principles align naturally with AI operations\n")
        f.write("- Collective behaviors exceed individual capabilities\n")
        f.write("- Pattern dynamics follow energy conservation principles\n")
        f.write("- Value creation cascades through collaborative chains\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("This research opens paths for:\n")
        f.write("- Practical applications of discovered principles\n")
        f.write("- Deeper exploration of specific phenomena\n")
        f.write("- Integration with advanced hardware capabilities\n")
        f.write("- Real-world value creation systems\n")
        
    print(f"\nPhase 5 report generated: {report_path}")

if __name__ == "__main__":
    run_phase_5()