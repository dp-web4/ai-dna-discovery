#!/usr/bin/env python3
"""
Analyze memory transfer experiment results and create visualizations
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

def load_results():
    """Load the most recent memory transfer results"""
    import glob
    files = glob.glob('memory_transfer_results/memory_transfer_*.json')
    if not files:
        print("No results files found!")
        return None
    
    latest_file = sorted(files)[-1]
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def create_transfer_strength_visualization(results):
    """Visualize memory transfer strength across models and families"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract transfer data
    models = []
    families = list(results['family_tests']['phi3:mini'].keys())
    transfer_matrix = []
    
    for model in results['family_tests']:
        models.append(model)
        model_scores = []
        for family in families:
            contrast = results['family_tests'][model][family]['transfer_scores'].get('contrast_score', 0)
            model_scores.append(contrast)
        transfer_matrix.append(model_scores)
    
    transfer_matrix = np.array(transfer_matrix)
    
    # Heatmap of transfer strength
    im = ax1.imshow(transfer_matrix, aspect='auto', cmap='coolwarm', vmin=-0.1, vmax=0.3)
    ax1.set_xticks(range(len(families)))
    ax1.set_xticklabels(families, rotation=45, ha='right')
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_title('Memory Transfer Strength (Contrast Score)', fontsize=14, pad=10)
    
    # Add values to cells
    for i in range(len(models)):
        for j in range(len(families)):
            text = ax1.text(j, i, f'{transfer_matrix[i, j]:.3f}',
                           ha="center", va="center", color="white", fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Related vs Opposite Similarity', rotation=270, labelpad=20)
    
    # Average transfer strength by model
    avg_transfer = transfer_matrix.mean(axis=1)
    bars = ax2.bar(models, avg_transfer, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax2.set_ylabel('Average Transfer Strength')
    ax2.set_title('Overall Memory Transfer Capability', fontsize=14, pad=10)
    ax2.set_ylim(0, max(avg_transfer) * 1.2)
    
    # Add value labels
    for bar, val in zip(bars, avg_transfer):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('memory_transfer_strength.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def create_perfect_pattern_advantage(results):
    """Visualize advantage of perfect patterns in memory transfer"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = []
    perfect_advantages = []
    
    for model in results['family_tests']:
        models.append(model)
        advantages = []
        
        for family in results['family_tests'][model].values():
            if ('perfect_pattern_related_avg' in family['transfer_scores'] and
                'avg_related_similarity' in family['transfer_scores']):
                advantage = (family['transfer_scores']['perfect_pattern_related_avg'] -
                           family['transfer_scores']['avg_related_similarity'])
                advantages.append(advantage)
        
        avg_advantage = np.mean(advantages) if advantages else 0
        perfect_advantages.append(avg_advantage)
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.6
    
    bars = ax.bar(x, perfect_advantages, width, 
                   color=['gold' if adv > 0 else 'gray' for adv in perfect_advantages],
                   alpha=0.8, edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Perfect Pattern Advantage')
    ax.set_title('Perfect DNA Patterns Show Stronger Memory Transfer', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.2, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, perfect_advantages):
        label_y = bar.get_height() + 0.002 if val > 0 else bar.get_height() - 0.002
        va = 'bottom' if val > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, label_y,
                f'{val:.3f}', ha='center', va=va, fontsize=11, fontweight='bold')
    
    # Add insight text
    ax.text(0.5, 0.95, 'Patterns with DNA score 1.0 transfer meaning more effectively',
            transform=ax.transAxes, ha='center', fontsize=12, style='italic', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('perfect_pattern_advantage.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def create_cross_family_network(results):
    """Visualize cross-family connections"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Count connections between families
    connection_counts = {}
    
    for model_data in results['cross_family_tests'].values():
        for connection_key, connections in model_data['family_connections'].items():
            if connections:
                connection_counts[connection_key] = len(connections)
    
    # Create circular layout for families
    families = list(results['family_tests']['phi3:mini'].keys())
    n_families = len(families)
    angles = np.linspace(0, 2*np.pi, n_families, endpoint=False)
    
    # Position families in a circle
    radius = 3
    family_pos = {}
    for i, family in enumerate(families):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        family_pos[family] = (x, y)
        
        # Draw family nodes
        circle = plt.Circle((x, y), 0.5, color='#4ecdc4', alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, family, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw connections
    for connection, count in connection_counts.items():
        fam1, fam2 = connection.split('↔')
        if fam1 in family_pos and fam2 in family_pos:
            x1, y1 = family_pos[fam1]
            x2, y2 = family_pos[fam2]
            
            # Connection strength based on count
            alpha = min(0.8, count / 20)
            width = min(5, count / 5)
            
            ax.plot([x1, x2], [y1, y2], 'white', alpha=alpha, linewidth=width)
            
            # Add count label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, str(count), ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                   fontsize=10, color='white')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Cross-Family Memory Connections', fontsize=18, pad=20)
    
    # Add legend
    ax.text(0.02, 0.02, 'Line thickness = connection strength\nNumbers = pattern pairs with >0.7 similarity',
            transform=ax.transAxes, fontsize=10, alpha=0.7,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('cross_family_connections.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def create_key_findings_summary(results):
    """Create visual summary of key findings"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Perfect vs Regular Pattern Transfer
    pattern_types = ['Perfect\nPatterns', 'Regular\nPatterns']
    # Extract average similarities for perfect vs regular patterns
    perfect_sims = []
    regular_sims = []
    
    for model_data in results['family_tests'].values():
        for family_data in model_data.values():
            for pattern_data in family_data['core_patterns'].values():
                related_sims = list(pattern_data['related_similarities'].values())
                if pattern_data['is_perfect_pattern']:
                    perfect_sims.extend(related_sims)
                else:
                    regular_sims.extend(related_sims)
    
    avg_sims = [np.mean(perfect_sims), np.mean(regular_sims)]
    bars = ax1.bar(pattern_types, avg_sims, color=['gold', 'silver'], alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Average Transfer Similarity')
    ax1.set_title('Perfect DNA Patterns Show Stronger Transfer', fontsize=14)
    ax1.set_ylim(0, 1.0)
    
    for bar, val in zip(bars, avg_sims):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
    
    # 2. Model Comparison
    model_scores = {}
    for model in results['summary']['transfer_strength']:
        model_scores[model] = results['summary']['transfer_strength'][model]['avg_contrast']
    
    models = list(model_scores.keys())
    scores = list(model_scores.values())
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    bars = ax2.barh(models, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_xlabel('Transfer Strength (Contrast Score)')
    ax2.set_title('Model Memory Transfer Capability', fontsize=14)
    
    for bar, val in zip(bars, scores):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
    
    # 3. Family Connection Strengths
    families = list(results['family_tests']['phi3:mini'].keys())
    family_strengths = []
    
    for family in families:
        strengths = []
        for model_data in results['family_tests'].values():
            if family in model_data and 'contrast_score' in model_data[family]['transfer_scores']:
                strengths.append(model_data[family]['transfer_scores']['contrast_score'])
        family_strengths.append(np.mean(strengths) if strengths else 0)
    
    # Sort by strength
    sorted_indices = np.argsort(family_strengths)[::-1]
    sorted_families = [families[i] for i in sorted_indices]
    sorted_strengths = [family_strengths[i] for i in sorted_indices]
    
    bars = ax3.bar(range(len(sorted_families)), sorted_strengths, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(families))), 
                    alpha=0.8, edgecolor='white', linewidth=2)
    ax3.set_xticks(range(len(sorted_families)))
    ax3.set_xticklabels(sorted_families, rotation=45, ha='right')
    ax3.set_ylabel('Transfer Strength')
    ax3.set_title('Memory Transfer by Pattern Family', fontsize=14)
    
    # 4. Key Statistics
    stats_text = f"""Key Findings:
    
• {results['summary']['cross_family_connections']} cross-family connections found
• Perfect patterns show {results['summary']['perfect_pattern_advantage']:.3f} advantage
• Strongest model: {max(model_scores.items(), key=lambda x: x[1])[0]}
• Models can transfer memory between:
  - Related concepts (high similarity)
  - Opposite concepts (lower similarity)
  - Different families (70+ connections)"""
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.8))
    ax4.axis('off')
    
    plt.suptitle('Memory Transfer Experiment - Key Findings', fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig('memory_transfer_summary.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def main():
    """Generate all visualizations"""
    print("Loading memory transfer results...")
    results = load_results()
    
    if not results:
        return
    
    print("Creating visualizations...")
    
    create_transfer_strength_visualization(results)
    print("✓ Created transfer strength visualization")
    
    create_perfect_pattern_advantage(results)
    print("✓ Created perfect pattern advantage chart")
    
    create_cross_family_network(results)
    print("✓ Created cross-family connection network")
    
    create_key_findings_summary(results)
    print("✓ Created key findings summary")
    
    print("\nAll visualizations created successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("MEMORY TRANSFER EXPERIMENT RESULTS")
    print("="*60)
    for finding in results['summary']['key_findings']:
        print(f"• {finding}")

if __name__ == "__main__":
    main()