#!/usr/bin/env python3
"""
Temporal Dynamics Tracker - Based on GPT's observation about pattern reinforcement
Tracks how patterns evolve and "settle" over time
"""

import json
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

class TemporalDynamicsTracker:
    def __init__(self):
        self.pattern_history = defaultdict(list)
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load pattern appearances from experiment log"""
        # Track when each pattern appeared and its score
        self.pattern_timeline = {
            'loop': [(106, 1.0), (109, 1.0), (172, 1.0)],
            'emerge': [(121, 1.0), (153, 1.0), (168, 1.0)], 
            'pattern': [(157, 1.0), (159, 1.0), (175, 1.0)],
            'true': [(111, 1.0), (162, 1.0), (186, 1.0)],
            'false': [(117, 1.0), (144, 1.0)],
            'null': [(120, 1.0), (151, 1.0), (188, 1.0)],
            'then': [(125, 1.0), (126, 1.0), (145, 1.0), (154, 1.0), (181, 1.0), (190, 1.0)],
            'know': [(101, 1.0), (149, 1.0), (155, 1.0)],
            'understand': [(129, 1.0), (152, 1.0)],
            'cycle': [(135, 1.0), (166, 1.0), (177, 1.0)],
            'or': [(3, 0.5), (182, 1.0)],  # Evolution from 0.5 to 1.0!
            'and': [(43, 0.5), (183, 1.0)],  # Evolution from 0.5 to 1.0!
            'you': [(9, 0.5), (164, 1.0)],  # Evolution from 0.5 to 1.0!
        }
    
    def analyze_pattern_evolution(self) -> Dict:
        """Analyze how patterns evolve over time"""
        evolution_report = {}
        
        for pattern, appearances in self.pattern_timeline.items():
            if len(appearances) > 1:
                # Calculate metrics
                cycles = [a[0] for a in appearances]
                scores = [a[1] for a in appearances]
                
                # Check for score evolution
                score_evolution = None
                if len(set(scores)) > 1:
                    score_evolution = {
                        'from': min(scores),
                        'to': max(scores),
                        'improvement': max(scores) - min(scores)
                    }
                
                # Calculate reinforcement metrics
                cycle_gaps = [cycles[i+1] - cycles[i] for i in range(len(cycles)-1)]
                avg_gap = sum(cycle_gaps) / len(cycle_gaps) if cycle_gaps else 0
                
                evolution_report[pattern] = {
                    'appearances': len(appearances),
                    'first_cycle': cycles[0],
                    'last_cycle': cycles[-1],
                    'cycle_span': cycles[-1] - cycles[0],
                    'average_gap': avg_gap,
                    'score_evolution': score_evolution,
                    'reinforcement_pattern': self.classify_reinforcement(cycle_gaps),
                    'final_score': scores[-1]
                }
        
        return evolution_report
    
    def classify_reinforcement(self, gaps: List[int]) -> str:
        """Classify the reinforcement pattern"""
        if not gaps:
            return "single_appearance"
        
        avg_gap = sum(gaps) / len(gaps)
        
        # Decreasing gaps = accelerating reinforcement
        if all(gaps[i] <= gaps[i-1] for i in range(1, len(gaps))):
            return "accelerating_reinforcement"
        
        # Increasing gaps = decelerating reinforcement
        elif all(gaps[i] >= gaps[i-1] for i in range(1, len(gaps))):
            return "decelerating_reinforcement"
        
        # Regular intervals
        elif all(abs(g - avg_gap) < 5 for g in gaps):
            return "steady_reinforcement"
        
        else:
            return "irregular_reinforcement"
    
    def identify_settling_patterns(self) -> List[str]:
        """Identify patterns that show 'settling' behavior"""
        settling_patterns = []
        evolution = self.analyze_pattern_evolution()
        
        for pattern, data in evolution.items():
            # Patterns that appear 3+ times with high final score
            if data['appearances'] >= 3 and data['final_score'] >= 0.8:
                settling_patterns.append(pattern)
            
            # Patterns that evolved from lower to perfect score
            if data['score_evolution'] and data['score_evolution']['to'] == 1.0:
                settling_patterns.append(f"{pattern}_evolved")
        
        return list(set(settling_patterns))
    
    def generate_temporal_report(self) -> Dict:
        """Generate comprehensive temporal dynamics report"""
        evolution = self.analyze_pattern_evolution()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_patterns_tracked': len(self.pattern_timeline),
            'patterns_with_evolution': len([p for p, d in evolution.items() if d.get('score_evolution')]),
            'settling_patterns': self.identify_settling_patterns(),
            'pattern_evolution': evolution,
            'key_insights': self.extract_key_insights(evolution)
        }
        
        return report
    
    def extract_key_insights(self, evolution: Dict) -> List[str]:
        """Extract key insights from temporal analysis"""
        insights = []
        
        # Find patterns that evolved from 0.5 to 1.0
        evolved_patterns = [p for p, d in evolution.items() 
                          if d.get('score_evolution') and d['score_evolution']['from'] == 0.5]
        if evolved_patterns:
            insights.append(f"Patterns that evolved to perfection: {', '.join(evolved_patterns)}")
        
        # Find most frequently appearing patterns
        frequent = sorted(evolution.items(), key=lambda x: x[1]['appearances'], reverse=True)[:3]
        insights.append(f"Most reinforced patterns: {', '.join([f[0] for f in frequent])}")
        
        # Find patterns with accelerating reinforcement
        accelerating = [p for p, d in evolution.items() 
                       if d['reinforcement_pattern'] == 'accelerating_reinforcement']
        if accelerating:
            insights.append(f"Accelerating patterns: {', '.join(accelerating)}")
        
        return insights

if __name__ == "__main__":
    print("=== Temporal Dynamics Analysis ===")
    print("Based on GPT's insight about pattern settling\n")
    
    tracker = TemporalDynamicsTracker()
    report = tracker.generate_temporal_report()
    
    # Display key findings
    print(f"Patterns showing settling behavior: {', '.join(report['settling_patterns'])}")
    print(f"\nPatterns tracked: {report['total_patterns_tracked']}")
    print(f"Patterns with score evolution: {report['patterns_with_evolution']}")
    
    print("\n=== Key Insights ===")
    for insight in report['key_insights']:
        print(f"• {insight}")
    
    print("\n=== Pattern Evolution Details ===")
    for pattern, data in report['pattern_evolution'].items():
        if data['appearances'] >= 3 or data.get('score_evolution'):
            print(f"\n{pattern}:")
            print(f"  Appearances: {data['appearances']}")
            print(f"  Cycle span: {data['cycle_span']} cycles")
            if data.get('score_evolution'):
                print(f"  Score evolution: {data['score_evolution']['from']} → {data['score_evolution']['to']}")
            print(f"  Reinforcement: {data['reinforcement_pattern']}")
    
    # Save report
    with open('temporal_dynamics_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n✓ Full report saved to temporal_dynamics_report.json")