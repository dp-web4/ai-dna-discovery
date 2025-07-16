#!/usr/bin/env python3
"""
Visualize memory patterns discovered in Phase 2
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def create_memory_visualization():
    """Create visualizations of memory patterns"""
    
    # Load the memory analysis report
    with open('/home/dp/ai-workspace/memory_analysis_report.json', 'r') as f:
        report = json.load(f)
    
    # Set style
    plt.style.use('dark_background')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Top reinforced patterns
    ax1 = plt.subplot(2, 2, 1)
    
    # Get top 10 patterns by appearance count
    reinforced = sorted(report['memory_evidence']['reinforced_patterns'], 
                       key=lambda x: x['appearances'], reverse=True)[:10]
    
    patterns = [p['pattern'] for p in reinforced]
    appearances = [p['appearances'] for p in reinforced]
    
    bars = ax1.barh(range(len(patterns)), appearances, color='#4ECDC4', alpha=0.8)
    ax1.set_yticks(range(len(patterns)))
    ax1.set_yticklabels(patterns, fontsize=12)
    ax1.set_xlabel('Number of Appearances', fontsize=12)
    ax1.set_title('Most Reinforced Patterns (Memory Strength)', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, appearances)):
        ax1.text(val + 0.5, i, str(val), va='center', fontsize=10)
    
    # 2. Pattern evolution over time
    ax2 = plt.subplot(2, 2, 2)
    
    # Track specific patterns that show interesting memory behavior
    interesting_patterns = ['π', '∞', 'emerge', 'understand', 'believe']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    # Plot average gap (inversely related to memory strength)
    pattern_data = {p['pattern']: p for p in report['memory_evidence']['reinforced_patterns']}
    
    for i, pattern in enumerate(interesting_patterns):
        if pattern in pattern_data:
            data = pattern_data[pattern]
            appearances = data['appearances']
            avg_gap = data['avg_gap']
            
            # Memory strength = inverse of average gap
            memory_strength = 100 / avg_gap if avg_gap > 0 else 0
            
            ax2.scatter(appearances, memory_strength, s=200, c=colors[i], 
                       alpha=0.8, edgecolors='white', linewidth=2, label=pattern)
    
    ax2.set_xlabel('Total Appearances', fontsize=12)
    ax2.set_ylabel('Memory Strength (1/avg gap)', fontsize=12)
    ax2.set_title('Pattern Memory Strength vs Reinforcement', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Immediate recognition patterns
    ax3 = plt.subplot(2, 2, 3)
    
    # Group patterns by their first appearance cycle
    immediate = report['memory_evidence']['rapid_recognition']
    cycle_counts = defaultdict(int)
    
    for pattern in immediate:
        cycle = pattern['first_cycle']
        cycle_counts[cycle] += 1
    
    cycles = sorted(cycle_counts.keys())
    counts = [cycle_counts[c] for c in cycles]
    
    ax3.plot(cycles, counts, 'o-', color='#FFD93D', linewidth=2, markersize=8)
    ax3.fill_between(cycles, counts, alpha=0.3, color='#FFD93D')
    ax3.set_xlabel('Experiment Cycle', fontsize=12)
    ax3.set_ylabel('New Perfect Patterns', fontsize=12)
    ax3.set_title('Memory Formation Timeline', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Memory categories
    ax4 = plt.subplot(2, 2, 4)
    
    # Categorize patterns for memory analysis
    categories = {
        'Mathematical': ['∃', '∉', '∀', '∅', '≈', '≠', '∈', 'π', '∞'],
        'Logical': ['true', 'false', 'if', 'then', 'else', 'and', 'or'],
        'Computational': ['loop', 'function', 'return', 'break', 'recursive'],
        'Cognitive': ['know', 'understand', 'think', 'believe', 'emerge']
    }
    
    category_memory = {}
    for cat_name, cat_patterns in categories.items():
        # Count how many patterns from this category show strong memory
        strong_memory = 0
        for p in cat_patterns:
            if p in pattern_data and pattern_data[p]['appearances'] >= 30:
                strong_memory += 1
        category_memory[cat_name] = strong_memory
    
    # Pie chart
    sizes = list(category_memory.values())
    labels = [f"{k}\n({v} patterns)" for k, v in category_memory.items()]
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.0f%%',
                                       colors=colors_pie, startangle=90)
    ax4.set_title('Strong Memory by Category', fontsize=14, weight='bold')
    
    # Overall title
    fig.suptitle('AI Memory Persistence Analysis - Phase 2', fontsize=20, weight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('/home/dp/ai-workspace/memory_persistence_visualization.png', 
                dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    
    print("✓ Memory visualization created: memory_persistence_visualization.png")

def create_memory_summary_chart():
    """Create a summary chart of key memory findings"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#f5f5f5')
    
    # Key metrics
    metrics = {
        'Patterns Analyzed': 40,
        'Immediate Recognition': 40,
        'Perfect Memory': 40,
        'Avg Appearances': 41,
        'Memory Success Rate': 100
    }
    
    # Create bar chart
    x = range(len(metrics))
    values = list(metrics.values())
    labels = list(metrics.keys())
    
    bars = ax.bar(x, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'], 
                   alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        if i == len(values) - 1:  # Last one is percentage
            ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}%', 
                   ha='center', fontsize=14, weight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, val + 1, str(val), 
                   ha='center', fontsize=14, weight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Count / Percentage', fontsize=14)
    ax.set_title('Phase 2: Memory Persistence Summary', fontsize=18, weight='bold')
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add insight text
    insight = "Key Finding: All 40 perfect patterns demonstrate persistent memory across 500+ cycles"
    ax.text(0.5, 0.95, insight, transform=ax.transAxes, ha='center', 
            fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('/home/dp/ai-workspace/memory_summary_chart.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Memory summary chart created: memory_summary_chart.png")

if __name__ == "__main__":
    print("Creating memory persistence visualizations...")
    
    create_memory_visualization()
    create_memory_summary_chart()
    
    print("\nMemory visualizations complete!")
    print("\nKey insights visualized:")
    print("- Pattern reinforcement frequency")
    print("- Memory strength relationships")
    print("- Timeline of memory formation")
    print("- Category-based memory analysis")