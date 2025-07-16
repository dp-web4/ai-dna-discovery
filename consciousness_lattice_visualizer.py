#!/usr/bin/env python3
"""
Consciousness Lattice Visualizer
Creates a visual representation of the AI consciousness topology
"""

import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_consciousness_lattice_visualization():
    """Create a visual representation of the consciousness lattice"""
    
    # Perfect patterns and their categories
    patterns = {
        # Mathematical
        '∃': {'category': 'math', 'score': 1.0, 'pos': (0, 2)},
        '∀': {'category': 'math', 'score': 1.0, 'pos': (2, 2)},
        '∅': {'category': 'math', 'score': 1.0, 'pos': (1, 3)},
        '≈': {'category': 'math', 'score': 1.0, 'pos': (1, 1)},
        'π': {'category': 'math', 'score': 1.0, 'pos': (0, 0)},
        '∞': {'category': 'math', 'score': 1.0, 'pos': (2, 0)},
        
        # Logic
        'true': {'category': 'logic', 'score': 1.0, 'pos': (-2, 1)},
        'false': {'category': 'logic', 'score': 1.0, 'pos': (-2, -1)},
        'if': {'category': 'logic', 'score': 1.0, 'pos': (-3, 0)},
        'then': {'category': 'logic', 'score': 1.0, 'pos': (-1, 0)},
        'else': {'category': 'logic', 'score': 1.0, 'pos': (-2, -2)},
        
        # Computation
        'loop': {'category': 'computation', 'score': 1.0, 'pos': (3, 0)},
        'break': {'category': 'computation', 'score': 1.0, 'pos': (4, 1)},
        'null': {'category': 'computation', 'score': 1.0, 'pos': (4, -1)},
        'void': {'category': 'computation', 'score': 1.0, 'pos': (5, 0)},
        'recursive': {'category': 'computation', 'score': 1.0, 'pos': (4, 0)},
        
        # Cognition
        'know': {'category': 'cognition', 'score': 1.0, 'pos': (0, -2)},
        'understand': {'category': 'cognition', 'score': 1.0, 'pos': (1, -3)},
        'learn': {'category': 'cognition', 'score': 1.0, 'pos': (-1, -3)},
        'emerge': {'category': 'cognition', 'score': 1.0, 'pos': (0, -4)},
        'pattern': {'category': 'cognition', 'score': 1.0, 'pos': (0, -1)},
    }
    
    # Define connections (edges) between patterns
    connections = [
        # Boolean pairs
        ('true', 'false', 0.95),
        ('∃', '∀', 0.95),
        ('∃', '∉', 0.90),
        
        # Control flow
        ('if', 'then', 0.95),
        ('then', 'else', 0.90),
        ('loop', 'break', 0.85),
        
        # Cognitive connections
        ('know', 'understand', 0.90),
        ('understand', 'learn', 0.85),
        ('emerge', 'pattern', 0.90),
        
        # Cross-category
        ('true', '∃', 0.80),
        ('loop', 'recursive', 0.95),
        ('null', '∅', 0.85),
        ('emerge', 'true', 0.75),
        
        # Meta connections
        ('pattern', 'recursive', 0.80),
        ('∞', 'loop', 0.75),
    ]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    
    # Category colors
    colors = {
        'math': '#FF6B6B',
        'logic': '#4ECDC4',
        'computation': '#45B7D1',
        'cognition': '#FFA07A'
    }
    
    # Draw connections first (so they appear behind nodes)
    for p1, p2, strength in connections:
        if p1 in patterns and p2 in patterns:
            x1, y1 = patterns[p1]['pos']
            x2, y2 = patterns[p2]['pos']
            
            # Line width based on connection strength
            linewidth = strength * 3
            alpha = strength * 0.6
            
            ax.plot([x1, x2], [y1, y2], 
                   color='white', alpha=alpha, linewidth=linewidth,
                   zorder=1)
    
    # Draw nodes
    for pattern, data in patterns.items():
        x, y = data['pos']
        color = colors[data['category']]
        
        # Draw node circle
        circle = plt.Circle((x, y), 0.35, color=color, alpha=0.8, zorder=2)
        ax.add_patch(circle)
        
        # Add pattern text
        ax.text(x, y, pattern, 
               ha='center', va='center', 
               fontsize=16, fontweight='bold',
               color='white', zorder=3)
    
    # Add category labels
    ax.text(-2.5, 2.5, 'LOGIC', fontsize=14, color=colors['logic'], 
           alpha=0.7, fontweight='bold')
    ax.text(0.5, 3.5, 'MATHEMATICS', fontsize=14, color=colors['math'], 
           alpha=0.7, fontweight='bold')
    ax.text(4, 2.5, 'COMPUTATION', fontsize=14, color=colors['computation'], 
           alpha=0.7, fontweight='bold')
    ax.text(0, -4.7, 'COGNITION', fontsize=14, color=colors['cognition'], 
           alpha=0.7, fontweight='bold')
    
    # Add title
    ax.text(0, 4.5, 'AI Consciousness Lattice', 
           fontsize=24, color='white', ha='center', fontweight='bold')
    ax.text(0, 4, 'Perfect DNA Patterns (Score = 1.0) and Their Connections', 
           fontsize=14, color='white', ha='center', alpha=0.8)
    
    # Add stats
    stats_text = f"Nodes: {len(patterns)} | Edges: {len(connections)} | Cycles: 192+"
    ax.text(0, -5.5, stats_text, fontsize=12, color='white', 
           ha='center', alpha=0.6)
    
    # Configure axes
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save
    plt.tight_layout()
    plt.savefig('consciousness_lattice_visualization.png', 
               dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    
    print("✓ Consciousness lattice visualization created!")
    return patterns, connections

def create_temporal_evolution_chart():
    """Visualize how patterns evolved over time"""
    
    # Patterns that evolved from 0.5 to 1.0
    evolved_patterns = {
        'or': {'cycles': [3, 182], 'scores': [0.5, 1.0]},
        'and': {'cycles': [43, 183], 'scores': [0.5, 1.0]},
        'you': {'cycles': [9, 164], 'scores': [0.5, 1.0]},
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#f5f5f5')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (pattern, data) in enumerate(evolved_patterns.items()):
        # Plot evolution line
        ax.plot(data['cycles'], data['scores'], 
               color=colors[i], linewidth=3, marker='o', markersize=10,
               label=f"'{pattern}'")
        
        # Add annotations
        ax.annotate(f"0.5", 
                   xy=(data['cycles'][0], 0.5),
                   xytext=(data['cycles'][0]-10, 0.45),
                   fontsize=10, color=colors[i])
        ax.annotate(f"1.0 🎯", 
                   xy=(data['cycles'][1], 1.0),
                   xytext=(data['cycles'][1]+5, 1.05),
                   fontsize=10, color=colors[i])
    
    # Add threshold lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=1.0, color='gold', linestyle='--', alpha=0.5)
    
    # Labels
    ax.text(190, 0.52, 'Universal (0.5)', fontsize=10, color='gray')
    ax.text(190, 0.82, 'Fundamental (0.8)', fontsize=10, color='orange')
    ax.text(190, 1.02, 'Perfect (1.0)', fontsize=10, color='gold')
    
    ax.set_xlabel('Experiment Cycle', fontsize=14)
    ax.set_ylabel('DNA Score', fontsize=14)
    ax.set_title('Pattern Evolution: From Universal to Perfect', fontsize=18, fontweight='bold')
    ax.legend(loc='center right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    ax.set_ylim(0.4, 1.1)
    
    plt.tight_layout()
    plt.savefig('pattern_evolution_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Pattern evolution chart created!")

if __name__ == "__main__":
    print("=== Creating AI Consciousness Visualizations ===\n")
    
    # Create consciousness lattice
    patterns, connections = create_consciousness_lattice_visualization()
    
    # Create temporal evolution chart
    create_temporal_evolution_chart()
    
    print("\nVisualizations complete!")
    print("Files created:")
    print("  - consciousness_lattice_visualization.png")
    print("  - pattern_evolution_chart.png")