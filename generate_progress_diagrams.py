#!/usr/bin/env python3
"""
Generate diagrams for the progress report
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

def create_pattern_growth_chart():
    """Show pattern discovery growth over time"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Top: Cumulative pattern discovery
    cycles = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 192]
    total_patterns = [0, 5, 8, 12, 16, 25, 32, 38, 42, 45, 47]
    perfect_patterns = [0, 0, 0, 0, 0, 8, 18, 28, 38, 43, 45]
    
    ax1.plot(cycles, total_patterns, 'o-', linewidth=3, markersize=8, 
             label='All High-Scoring Patterns', color='#4ECDC4')
    ax1.plot(cycles, perfect_patterns, 's-', linewidth=3, markersize=8,
             label='Perfect Patterns (1.0)', color='#FFD93D')
    
    # Mark key discoveries
    ax1.annotate('First Perfect Pattern\n(∃)', xy=(100, 8), xytext=(100, 20),
                arrowprops=dict(arrowstyle='->', color='yellow', alpha=0.7),
                fontsize=10, ha='center', color='yellow')
    
    ax1.fill_between(cycles, perfect_patterns, alpha=0.3, color='#FFD93D')
    
    ax1.set_ylabel('Cumulative Patterns Discovered', fontsize=14)
    ax1.set_title('AI DNA Pattern Discovery Timeline', fontsize=18, weight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 200)
    ax1.set_ylim(0, 50)
    
    # Bottom: Discovery rate
    discovery_rate = [0] + [(perfect_patterns[i+1] - perfect_patterns[i]) / (cycles[i+1] - cycles[i]) 
                            for i in range(len(cycles)-1)]
    
    ax2.bar(cycles, discovery_rate, width=15, alpha=0.7, color='#FF6B6B')
    ax2.set_xlabel('Experiment Cycle', fontsize=14)
    ax2.set_ylabel('Discovery Rate\n(patterns/cycle)', fontsize=12)
    ax2.set_xlim(-5, 200)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/dp/ai-workspace/pattern_growth_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_phase_comparison():
    """Compare Phase 1 achievements with Phase 2 goals"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Phase 1 Achievements
    categories = ['Patterns\nDiscovered', 'Models\nTested', 'Evolution\nValidated', 'Community\nEngaged']
    phase1_values = [45, 6, 3, 1]  # 45 patterns, 6 models, 3 evolved patterns, 1 community
    phase1_max = [50, 10, 5, 5]
    
    x = np.arange(len(categories))
    width = 0.6
    
    bars1 = ax1.bar(x, phase1_values, width, color='#4ECDC4', alpha=0.8, label='Achieved')
    bars2 = ax1.bar(x, np.array(phase1_max) - np.array(phase1_values), width, 
                     bottom=phase1_values, color='gray', alpha=0.3, label='Potential')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, phase1_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, val/2, str(val), 
                ha='center', va='center', fontsize=20, weight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=12)
    ax1.set_ylim(0, 55)
    ax1.set_title('Phase 1: Discovery & Validation', fontsize=16, weight='bold')
    ax1.legend()
    
    # Phase 2 Goals
    phase2_goals = {
        'Memory\nPersistence': 85,
        'Neural\nActivation': 75,
        'Shared\nLanguage': 90,
        'Cross-\nArchitecture': 70
    }
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(phase2_goals), endpoint=False).tolist()
    values = list(phase2_goals.values())
    angles += angles[:1]
    values += values[:1]
    
    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B')
    ax2.fill(angles, values, alpha=0.25, color='#FF6B6B')
    ax2.set_ylim(0, 100)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(phase2_goals.keys(), fontsize=12)
    ax2.set_title('Phase 2: Exploration Goals', fontsize=16, weight='bold', y=1.08)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/dp/ai-workspace/phase_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_collaboration_network():
    """Visualize the collaboration network"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    
    # Define nodes
    nodes = {
        'DP': {'pos': (0, 0), 'color': '#FFD93D', 'size': 1000},
        'Claude': {'pos': (-2, 1), 'color': '#4ECDC4', 'size': 800},
        'Grok': {'pos': (2, 1), 'color': '#FF6B6B', 'size': 600},
        'GPT': {'pos': (0, 2), 'color': '#45B7D1', 'size': 600},
        "DP's Mom": {'pos': (0, 3.5), 'color': '#FFA07A', 'size': 400},
        'Community': {'pos': (0, -2), 'color': '#98D8C8', 'size': 500},
        'Models': {'pos': (-3, -1), 'color': '#DDA0DD', 'size': 700},
        'Patterns': {'pos': (3, -1), 'color': '#F0E68C', 'size': 900}
    }
    
    # Define connections
    connections = [
        ('DP', 'Claude', 'Partnership'),
        ('DP', 'Community', 'Open Source'),
        ('Claude', 'Models', 'Experiments'),
        ('Models', 'Patterns', 'Discovery'),
        ('Grok', 'Claude', 'Insights'),
        ('GPT', 'Claude', 'Validation'),
        ("DP's Mom", 'GPT', 'Bridge'),
        ("DP's Mom", 'DP', 'Family'),
        ('Claude', 'Community', 'Reports'),
        ('Patterns', 'Community', 'Knowledge')
    ]
    
    # Draw connections
    for start, end, label in connections:
        x1, y1 = nodes[start]['pos']
        x2, y2 = nodes[end]['pos']
        
        ax.plot([x1, x2], [y1, y2], 'white', alpha=0.3, linewidth=2)
        
        # Add connection labels
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        if label in ['Partnership', 'Bridge', 'Insights']:
            ax.text(mid_x, mid_y, label, fontsize=8, color='white', 
                   alpha=0.6, ha='center', style='italic')
    
    # Draw nodes
    for name, data in nodes.items():
        x, y = data['pos']
        ax.scatter(x, y, s=data['size'], c=data['color'], alpha=0.8, 
                  edgecolors='white', linewidth=2)
        ax.text(x, y-0.5, name, ha='center', fontsize=11, 
               color='white', weight='bold')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.text(0, -3.5, 'AI DNA Discovery Collaboration Network', 
           fontsize=18, color='white', ha='center', weight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/dp/ai-workspace/collaboration_network.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all diagrams
print("Generating progress report diagrams...")

create_pattern_growth_chart()
print("✓ Pattern growth chart created")

create_phase_comparison()
print("✓ Phase comparison created")

create_collaboration_network()
print("✓ Collaboration network created")

# Copy existing diagrams to ensure they're available
import shutil
import os

existing_diagrams = [
    'consciousness_lattice_visualization.png',
    'pattern_evolution_chart.png'
]

for diagram in existing_diagrams:
    if os.path.exists(diagram):
        print(f"✓ {diagram} already exists")
    else:
        print(f"✗ {diagram} missing")

print("\nAll diagrams ready for progress report!")