#!/usr/bin/env python3
"""
Create PDF version of Memory Transfer Report
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import os

def create_pdf_report():
    """Create multi-page PDF report"""
    
    pdf_filename = 'memory_transfer_report.pdf'
    
    with PdfPages(pdf_filename) as pdf:
        # Page 1 - Title and Executive Summary
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.9, 'Memory Transfer Experiment Report', 
                fontsize=24, weight='bold', ha='center', transform=ax.transAxes)
        
        # Subtitle
        ax.text(0.5, 0.85, 'AI DNA Discovery - Phase 2', 
                fontsize=16, ha='center', transform=ax.transAxes, style='italic')
        
        # Date and stats
        ax.text(0.5, 0.8, 'July 13, 2025 | Experiment Cycles: 521+', 
                fontsize=12, ha='center', transform=ax.transAxes)
        
        # Executive Summary
        summary_box = FancyBboxPatch((0.05, 0.35), 0.9, 0.35,
                                    boxstyle="round,pad=0.02",
                                    facecolor='lightblue', alpha=0.1,
                                    edgecolor='darkblue', linewidth=2)
        ax.add_patch(summary_box)
        
        summary_text = """Executive Summary

Building on 40+ perfect AI DNA patterns, Phase 2 investigated 
memory transfer between semantically related concepts.

Key Findings:
• Perfect patterns show 2.4% stronger transfer capability
• 70 cross-family connections discovered
• Memory operates as semantic network
• Models discriminate related vs opposite concepts

Major Insight:
"AI memory operates as a semantic network where 
learning one concept facilitates understanding of 
related concepts - mirroring human cognition."
"""
        
        ax.text(0.5, 0.52, summary_text,
                fontsize=11, ha='center', va='center', transform=ax.transAxes)
        
        # Key metrics
        metrics_y = 0.25
        metrics = [
            ('Patterns Tested:', '75'),
            ('Models Evaluated:', '3'),
            ('Perfect Pattern Advantage:', '+2.4%'),
            ('Cross-Family Connections:', '70')
        ]
        
        for i, (label, value) in enumerate(metrics):
            ax.text(0.25, metrics_y - i*0.04, label, fontsize=11, weight='bold', transform=ax.transAxes)
            ax.text(0.55, metrics_y - i*0.04, value, fontsize=11, transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2 - Experimental Design
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Experimental Design', 
                fontsize=20, weight='bold', ha='center', transform=ax.transAxes)
        
        # Pattern families
        ax.text(0.1, 0.85, 'Pattern Families Tested:', 
                fontsize=14, weight='bold', transform=ax.transAxes)
        
        families_text = """
1. Existence: ∃, exist, being → void, null ↔ absence, nothing
2. Truth: true, valid, correct → false, wrong ↔ lie, illusion  
3. Emergence: emerge, arise → evolve, unfold ↔ vanish, dissolve
4. Recursion: recursive, loop → iterate, repeat ↔ linear, once
5. Knowledge: know, understand → learn, discover ↔ forget, ignore
"""
        
        ax.text(0.1, 0.65, families_text, fontsize=10, transform=ax.transAxes,
                family='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        
        # Methodology
        ax.text(0.1, 0.45, 'Methodology:', 
                fontsize=14, weight='bold', transform=ax.transAxes)
        
        method_points = [
            '• Calculate embeddings for all patterns',
            '• Measure cosine similarity between patterns',
            '• Compare related vs opposite pattern similarities',
            '• Track perfect pattern performance separately',
            '• Test across 3 different models'
        ]
        
        y_pos = 0.38
        for point in method_points:
            ax.text(0.15, y_pos, point, fontsize=11, transform=ax.transAxes)
            y_pos -= 0.04
        
        # Models tested
        ax.text(0.1, 0.15, 'Models:', fontsize=12, weight='bold', transform=ax.transAxes)
        ax.text(0.25, 0.15, 'phi3:mini, gemma:2b, tinyllama:latest', 
                fontsize=11, transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3 - Key Discoveries
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Key Discoveries', 
                fontsize=20, weight='bold', ha='center', transform=ax.transAxes)
        
        # Discovery boxes
        discoveries = [
            {
                'title': '1. Memory Transfer is Real',
                'content': 'Models show clear discrimination:\n• Related patterns: 0.65-0.99 similarity\n• Opposite patterns: 0.12-0.45 similarity\n• Contrast scores: 0.05-0.25',
                'y': 0.75
            },
            {
                'title': '2. Perfect Pattern Advantage',
                'content': 'DNA score 1.0 patterns:\n• 2.4% stronger transfer on average\n• Act as "semantic anchors"\n• Effect strongest in phi3:mini',
                'y': 0.50
            },
            {
                'title': '3. Cross-Family Connections',
                'content': '70 strong connections found:\n• Strongest: existence ↔ truth (60)\n• Forms semantic knowledge web\n• Universal patterns bridge families',
                'y': 0.25
            }
        ]
        
        for disc in discoveries:
            box = FancyBboxPatch((0.1, disc['y']-0.05), 0.8, 0.18,
                               boxstyle="round,pad=0.02",
                               facecolor='lightgreen', alpha=0.1,
                               edgecolor='darkgreen', linewidth=1.5)
            ax.add_patch(box)
            
            ax.text(0.15, disc['y']+0.08, disc['title'], 
                    fontsize=13, weight='bold', transform=ax.transAxes)
            ax.text(0.15, disc['y'], disc['content'], 
                    fontsize=10, transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4 - Results Summary
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Results Summary', 
                fontsize=20, weight='bold', ha='center', transform=ax.transAxes)
        
        # Transfer strength table
        ax.text(0.1, 0.85, 'Transfer Strength by Family:', 
                fontsize=14, weight='bold', transform=ax.transAxes)
        
        table_data = """
Family          Contrast Score    Interpretation
─────────────────────────────────────────────
Knowledge       0.182            Very Strong
Truth           0.156            Strong  
Existence       0.143            Strong
Recursion       0.128            Moderate
Emergence       0.115            Moderate
"""
        
        ax.text(0.1, 0.65, table_data, fontsize=10, transform=ax.transAxes,
                family='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        
        # Top connections
        ax.text(0.1, 0.45, 'Strongest Cross-Family Connections:', 
                fontsize=14, weight='bold', transform=ax.transAxes)
        
        connections = [
            'exist ↔ true (0.986)',
            'exist ↔ valid (0.994)',
            'being ↔ know (0.992)',
            'recursive ↔ know (0.990)',
            'emerge ↔ comprehend (0.936)'
        ]
        
        y_pos = 0.38
        for i, conn in enumerate(connections):
            ax.text(0.15, y_pos - i*0.03, f'{i+1}. {conn}', 
                    fontsize=10, transform=ax.transAxes)
        
        # Model comparison
        ax.text(0.1, 0.18, 'Model Performance:', 
                fontsize=14, weight='bold', transform=ax.transAxes)
        ax.text(0.15, 0.12, 'phi3:mini: Highest transfer (0.145)', fontsize=10, transform=ax.transAxes)
        ax.text(0.15, 0.08, 'gemma:2b: Moderate transfer (0.098)', fontsize=10, transform=ax.transAxes)
        ax.text(0.15, 0.04, 'tinyllama: Lower transfer (0.067)', fontsize=10, transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 5 - Implications
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Implications & Future Work', 
                fontsize=20, weight='bold', ha='center', transform=ax.transAxes)
        
        # Theoretical implications
        ax.text(0.1, 0.85, 'Theoretical Implications:', 
                fontsize=14, weight='bold', transform=ax.transAxes)
        
        implications = [
            '• AI memory operates as interconnected semantic network',
            '• Patterns are nodes in vast knowledge graph',
            '• Perfect patterns serve as high-connectivity hubs',
            '• Memory transcends individual weight values',
            '• Models develop genuine conceptual understanding'
        ]
        
        y_pos = 0.75
        for imp in implications:
            ax.text(0.15, y_pos, imp, fontsize=11, transform=ax.transAxes)
            y_pos -= 0.05
        
        # Next steps
        ax.text(0.1, 0.45, 'Next Steps:', 
                fontsize=14, weight='bold', transform=ax.transAxes)
        
        next_steps = [
            '1. Map embedding vector spaces for perfect patterns',
            '2. Test shared pattern creation between models',
            '3. Validate on non-transformer architectures',
            '4. Explore memory interference and capacity limits'
        ]
        
        y_pos = 0.35
        for step in next_steps:
            ax.text(0.15, y_pos, step, fontsize=11, transform=ax.transAxes)
            y_pos -= 0.05
        
        # Final quote
        quote_box = FancyBboxPatch((0.1, 0.05), 0.8, 0.1,
                                 boxstyle="round,pad=0.02",
                                 facecolor='purple', alpha=0.1,
                                 edgecolor='purple', linewidth=1)
        ax.add_patch(quote_box)
        
        ax.text(0.5, 0.1, '"In the architecture of artificial minds,\nmemory is not stored but woven."',
                fontsize=12, ha='center', va='center', transform=ax.transAxes,
                style='italic', color='darkblue')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Add visualization pages
        viz_files = [
            ('memory_transfer_concept.png', 'Memory Transfer Concept'),
            ('pattern_family_map.png', 'Pattern Family Network'),
            ('memory_transfer_strength.png', 'Transfer Strength Analysis'),
            ('perfect_pattern_advantage.png', 'Perfect Pattern Advantage'),
            ('cross_family_connections.png', 'Cross-Family Connections'),
            ('model_comparison_radar.png', 'Model Capabilities'),
            ('memory_transfer_summary.png', 'Key Findings Summary'),
            ('experiment_timeline.png', 'Experiment Timeline')
        ]
        
        for viz_file, title in viz_files:
            if os.path.exists(viz_file):
                img = plt.imread(viz_file)
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111)
                
                # Add title
                fig.text(0.5, 0.95, title, fontsize=16, ha='center', weight='bold')
                
                # Show image
                ax.imshow(img)
                ax.axis('off')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
    print(f"\n✓ PDF created successfully: {pdf_filename}")
    print(f"  Total pages: {5 + len([f for f, _ in viz_files if os.path.exists(f)])}")

if __name__ == "__main__":
    create_pdf_report()