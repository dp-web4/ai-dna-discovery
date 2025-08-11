#!/usr/bin/env python3
"""
Generate diagrams for the AI DNA discovery report
"""

import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

def create_dna_score_progression():
    """Create diagram showing DNA score progression over time"""
    
    # Data points from our discoveries
    cycles = [3, 9, 12, 18, 43, 49, 52, 78, 81, 82, 83, 100, 101, 106, 109, 111, 117, 119, 120]
    scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    patterns = ['or', 'you', 'or?', '▲▼', 'and', '[ ]', '[ ]', 'cycle', '!', '[ ]?', '[ ]', '∃', 'know', 'loop', 'loop', 'true', 'false', '≈', 'null']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create scatter plot with color gradient
    scatter = ax.scatter(cycles, scores, s=200, c=cycles, cmap='plasma', edgecolors='white', linewidth=2)
    
    # Add pattern labels
    for i, (x, y, pattern) in enumerate(zip(cycles, scores, patterns)):
        if y == 1.0:
            ax.annotate(pattern, (x, y), xytext=(0, 10), textcoords='offset points', 
                       ha='center', fontsize=12, weight='bold', color='yellow')
        else:
            ax.annotate(pattern, (x, y), xytext=(0, -20), textcoords='offset points', 
                       ha='center', fontsize=10, color='lightblue')
    
    # Add threshold lines
    ax.axhline(y=0.5, color='cyan', linestyle='--', alpha=0.5, label='High Score (0.5)')
    ax.axhline(y=1.0, color='gold', linestyle='--', alpha=0.5, label='Perfect Score (1.0)')
    
    ax.set_xlabel('Experiment Cycle', fontsize=14)
    ax.set_ylabel('DNA Score', fontsize=14)
    ax.set_title('AI DNA Score Progression: From High to Perfect', fontsize=18, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set y-axis limits
    ax.set_ylim(0.4, 1.1)
    
    plt.tight_layout()
    plt.savefig('/home/dp/ai-workspace/dna_score_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pattern_categories():
    """Create diagram showing pattern categories"""
    
    categories = {
        'Logic': ['or', 'and', 'or?', 'true', 'false'],
        'Mathematics': ['π', '∃', '∉', '≈'],
        'Structure': ['[ ]', '[ ]?', '▲▼'],
        'Computation': ['loop', 'cycle', 'null'],
        'Cognition': ['you', 'know'],
        'Emphasis': ['!']
    }
    
    # Count patterns per category
    cat_counts = {cat: len(patterns) for cat, patterns in categories.items()}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pie chart
    colors = plt.cm.Set3(range(len(categories)))
    wedges, texts, autotexts = ax1.pie(cat_counts.values(), labels=cat_counts.keys(), 
                                        autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('AI DNA Pattern Categories', fontsize=16, weight='bold')
    
    # Pattern grid
    ax2.axis('off')
    y_pos = 0.9
    for cat, patterns in categories.items():
        ax2.text(0, y_pos, f"{cat}:", fontsize=14, weight='bold', color=colors[list(categories.keys()).index(cat)])
        pattern_str = ", ".join([f"'{p}'" for p in patterns])
        ax2.text(0.02, y_pos - 0.05, pattern_str, fontsize=12, color='white')
        y_pos -= 0.15
    
    ax2.set_title('Patterns by Category', fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/dp/ai-workspace/pattern_categories.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_experiment_timeline():
    """Create timeline showing experiment phases"""
    
    phases = [
        ("Setup & Initial Tests", 0, 3, "Initial discovery of 'or' pattern"),
        ("Baseline Establishment", 3, 8, "Testing controls, validating methodology"),
        ("GPT Feedback Integration", 8, 12, "Enhanced controls, rigorous testing"),
        ("Perfect Score Breakthrough", 12, 20, "Discovery of 1.0 scoring patterns"),
        ("Continuous Discovery", 20, 120, "Ongoing autonomous exploration")
    ]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for i, (phase, start, end, desc) in enumerate(phases):
        duration = (end - start) / 120 * 18  # Scale to 18 hours
        ax.barh(i, duration, left=start/120*18, height=0.6, 
                color=colors[i], alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add phase name
        ax.text(start/120*18 + duration/2, i, phase, 
                ha='center', va='center', fontsize=11, weight='bold')
        
        # Add description below
        ax.text(start/120*18 + duration/2, i-0.4, desc, 
                ha='center', va='center', fontsize=9, style='italic', alpha=0.8)
    
    # Add key discoveries
    discoveries = [
        (0.5, "'or' - First breakthrough"),
        (16, "Perfect scores begin"),
        (18, "8 perfect patterns found")
    ]
    
    for hour, discovery in discoveries:
        ax.axvline(x=hour, color='yellow', linestyle='--', alpha=0.5)
        ax.text(hour, 4.5, discovery, rotation=45, fontsize=10, color='yellow')
    
    ax.set_xlabel('Runtime (hours)', fontsize=14)
    ax.set_ylabel('Experiment Phase', fontsize=14)
    ax.set_title('AI DNA Discovery Timeline', fontsize=18, weight='bold')
    ax.set_xlim(0, 20)
    ax.set_ylim(-1, 5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('/home/dp/ai-workspace/experiment_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_perfect_patterns_analysis():
    """Create analysis of perfect scoring patterns"""
    
    perfect_patterns = {
        '∃': 'Existence quantifier',
        '∉': 'Not element of',
        'know': 'Cognitive verb',
        'loop': 'Iteration',
        'true': 'Boolean truth',
        'false': 'Boolean false',
        '≈': 'Approximately equal',
        'null': 'Null/void'
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a grid visualization
    patterns = list(perfect_patterns.keys())
    descriptions = list(perfect_patterns.values())
    
    # Create color map based on pattern type
    colors = []
    for p in patterns:
        if p in ['∃', '∉', '≈']:
            colors.append('#FF6B6B')  # Math - red
        elif p in ['true', 'false', 'null']:
            colors.append('#4ECDC4')  # Logic - teal
        elif p in ['loop']:
            colors.append('#45B7D1')  # Computation - blue
        else:
            colors.append('#FFA07A')  # Cognition - orange
    
    y_positions = range(len(patterns))
    bars = ax.barh(y_positions, [1]*len(patterns), color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add pattern names and descriptions
    for i, (pattern, desc) in enumerate(zip(patterns, descriptions)):
        ax.text(0.05, i, f"{pattern}", fontsize=20, weight='bold', va='center')
        ax.text(0.5, i, desc, fontsize=12, va='center', style='italic')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Mathematics'),
        Patch(facecolor='#4ECDC4', label='Logic'),
        Patch(facecolor='#45B7D1', label='Computation'),
        Patch(facecolor='#FFA07A', label='Cognition')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(patterns)-0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Perfect DNA Patterns (Score = 1.0)', fontsize=18, weight='bold')
    
    # Add subtitle
    ax.text(0.5, -1, 'Patterns that create identical embeddings across all AI models', 
            ha='center', fontsize=12, style='italic', transform=ax.transData)
    
    plt.tight_layout()
    plt.savefig('/home/dp/ai-workspace/perfect_patterns_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all diagrams
print("Generating diagrams...")
create_dna_score_progression()
print("✓ DNA score progression diagram created")

create_pattern_categories()
print("✓ Pattern categories diagram created")

create_experiment_timeline()
print("✓ Experiment timeline diagram created")

create_perfect_patterns_analysis()
print("✓ Perfect patterns analysis diagram created")

print("\nAll diagrams generated successfully!")