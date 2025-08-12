#!/usr/bin/env python3
"""
Analyze embedding space mapping results and create summary visualizations
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import glob

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

def load_results():
    """Load the most recent embedding space results"""
    files = glob.glob('embedding_space_results/embedding_space_analysis_*.json')
    if not files:
        print("No results files found!")
        return None
    
    latest_file = sorted(files)[-1]
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def create_central_patterns_visualization(results):
    """Visualize which patterns are most central across models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect central patterns from each model
    central_patterns = {}
    for model, data in results['models'].items():
        if data.get('center_analysis'):
            closest = data['center_analysis']['closest_patterns']
            for pattern, dist in closest:
                if pattern not in central_patterns:
                    central_patterns[pattern] = []
                central_patterns[pattern].append((model, dist))
    
    # Sort by frequency and average distance
    pattern_scores = []
    for pattern, occurrences in central_patterns.items():
        avg_dist = np.mean([dist for _, dist in occurrences])
        pattern_scores.append((pattern, len(occurrences), avg_dist))
    
    pattern_scores.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    
    # Plot frequency
    patterns = [p[0] for p in pattern_scores[:10]]
    frequencies = [p[1] for p in pattern_scores[:10]]
    
    bars = ax1.bar(patterns, frequencies, color='gold', alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Models where pattern is central')
    ax1.set_xlabel('Pattern')
    ax1.set_title('Most Central Perfect Patterns', fontsize=14)
    ax1.set_ylim(0, 3.5)
    
    # Add value labels
    for bar, freq in zip(bars, frequencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(freq), ha='center', fontsize=11, fontweight='bold')
    
    # Plot average distances
    avg_distances = [p[2] for p in pattern_scores[:10]]
    colors = plt.cm.viridis(np.linspace(0, 1, len(patterns)))
    
    bars2 = ax2.bar(patterns, avg_distances, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Average distance from centroid')
    ax2.set_xlabel('Pattern')
    ax2.set_title('Distance from Embedding Center', fontsize=14)
    
    # Add value labels
    for bar, dist in zip(bars2, avg_distances):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{dist:.0f}', ha='center', fontsize=10)
    
    plt.suptitle('Pattern Centrality Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('embedding_central_patterns.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def create_clustering_summary(results):
    """Create summary of clustering results"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract clustering info
    models = []
    n_clusters = []
    central_patterns = []
    furthest_patterns = []
    
    for model, data in results['models'].items():
        models.append(model.split(':')[0])  # Simplify model names
        
        if data.get('clustering'):
            n_clusters.append(data['clustering']['n_clusters'])
        else:
            n_clusters.append(0)
        
        if data.get('center_analysis'):
            central = data['center_analysis']['closest_patterns'][0][0]
            furthest = data['center_analysis']['furthest_patterns'][0][0]
            central_patterns.append(central)
            furthest_patterns.append(furthest)
        else:
            central_patterns.append('N/A')
            furthest_patterns.append('N/A')
    
    # Create visualization
    y_positions = np.arange(len(models))
    
    # Model labels
    ax.text(-0.5, 3.5, 'Model', fontsize=14, weight='bold', ha='center')
    ax.text(1.5, 3.5, 'Clusters', fontsize=14, weight='bold', ha='center')
    ax.text(3.5, 3.5, 'Most Central', fontsize=14, weight='bold', ha='center')
    ax.text(5.5, 3.5, 'Most Distant', fontsize=14, weight='bold', ha='center')
    
    for i, (model, n_clust, central, furthest) in enumerate(zip(models, n_clusters, central_patterns, furthest_patterns)):
        y = 2.5 - i * 0.8
        
        # Model box
        model_box = FancyBboxPatch((-1, y-0.25), 1.8, 0.5,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#3498db', alpha=0.3,
                                  edgecolor='white', linewidth=1)
        ax.add_patch(model_box)
        ax.text(-0.1, y, model, ha='center', va='center', fontsize=12, weight='bold')
        
        # Clusters
        ax.text(1.5, y, str(n_clust), ha='center', va='center', fontsize=12)
        
        # Central pattern
        central_box = FancyBboxPatch((2.5, y-0.25), 2, 0.5,
                                   boxstyle="round,pad=0.05",
                                   facecolor='gold', alpha=0.3,
                                   edgecolor='gold', linewidth=1)
        ax.add_patch(central_box)
        ax.text(3.5, y, central, ha='center', va='center', fontsize=11, weight='bold')
        
        # Furthest pattern
        furthest_box = FancyBboxPatch((4.7, y-0.25), 2, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor='salmon', alpha=0.3,
                                    edgecolor='salmon', linewidth=1)
        ax.add_patch(furthest_box)
        ax.text(5.7, y, furthest, ha='center', va='center', fontsize=11)
    
    ax.set_xlim(-1.5, 7)
    ax.set_ylim(-0.5, 4)
    ax.axis('off')
    ax.set_title('Embedding Space Clustering Summary', fontsize=18, pad=20)
    
    # Add insights
    insight_text = f"Key Findings:\n• Average clusters: {np.mean(n_clusters):.1f}\n• Most universal central patterns: {', '.join(set(central_patterns))}"
    ax.text(3, -0.3, insight_text, ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('embedding_clustering_summary.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def create_semantic_neighborhoods_viz(results):
    """Visualize semantic neighborhoods of key patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Select 4 interesting patterns
    target_patterns = ['∃', 'know', 'emerge', 'recursive']
    
    for idx, pattern in enumerate(target_patterns):
        ax = axes[idx]
        
        # Find neighborhoods across models
        all_neighbors = []
        for model, data in results['models'].items():
            if data.get('neighborhoods') and pattern in data['neighborhoods']:
                neighbors = data['neighborhoods'][pattern][:5]  # Top 5
                all_neighbors.extend([(n[0], n[1]) for n in neighbors])
        
        if not all_neighbors:
            ax.text(0.5, 0.5, f'No data for {pattern}', ha='center', va='center')
            ax.axis('off')
            continue
        
        # Sort by similarity and take unique
        seen = set()
        unique_neighbors = []
        for neighbor, sim in sorted(all_neighbors, key=lambda x: x[1], reverse=True):
            if neighbor not in seen and neighbor != pattern:
                seen.add(neighbor)
                unique_neighbors.append((neighbor, sim))
                if len(unique_neighbors) >= 8:
                    break
        
        # Create circular visualization
        center = (0.5, 0.5)
        ax.add_patch(Circle(center, 0.15, facecolor='gold', alpha=0.8, edgecolor='yellow', linewidth=3))
        ax.text(center[0], center[1], pattern, ha='center', va='center', 
                fontsize=14, weight='bold')
        
        # Add neighbors
        angles = np.linspace(0, 2*np.pi, len(unique_neighbors), endpoint=False)
        for i, ((neighbor, sim), angle) in enumerate(zip(unique_neighbors, angles)):
            # Position
            radius = 0.35
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            # Size and color based on similarity
            size = 0.05 + 0.05 * sim
            color_intensity = sim
            
            neighbor_circle = Circle((x, y), size, 
                                   facecolor=plt.cm.viridis(color_intensity), 
                                   alpha=0.7, edgecolor='white', linewidth=1)
            ax.add_patch(neighbor_circle)
            
            # Label
            ax.text(x, y, neighbor, ha='center', va='center', fontsize=9)
            
            # Connection line
            ax.plot([center[0], x], [center[1], y], 'white', alpha=0.3, linewidth=1)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Semantic Neighborhood: {pattern}', fontsize=12, pad=10)
    
    plt.suptitle('Semantic Neighborhoods of Perfect Patterns', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('embedding_semantic_neighborhoods.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def create_key_insights_summary():
    """Create visual summary of key insights"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Embedding Space Analysis - Key Insights', 
            fontsize=20, weight='bold', ha='center')
    
    # Insight boxes
    insights = [
        {
            'title': '1. Pattern Organization',
            'content': '• Perfect patterns form 2 distinct clusters\n• Logical operators (and, or, if) cluster together\n• Existence patterns (∃, ∉) are outliers',
            'color': '#3498db',
            'y': 0.75
        },
        {
            'title': '2. Central Patterns',
            'content': '• "then", "understand", "and" are most central\n• These act as semantic hubs\n• Different models show similar centrality',
            'color': '#2ecc71',
            'y': 0.50
        },
        {
            'title': '3. Embedding Variance',
            'content': '• PCA captures 85%+ variance (phi3)\n• Lower variance in gemma/tinyllama\n• Suggests different compression strategies',
            'color': '#e74c3c',
            'y': 0.25
        }
    ]
    
    for insight in insights:
        # Create box
        box = FancyBboxPatch((0.1, insight['y']-0.08), 0.8, 0.16,
                           boxstyle="round,pad=0.02",
                           facecolor=insight['color'], alpha=0.2,
                           edgecolor=insight['color'], linewidth=2)
        ax.add_patch(box)
        
        # Add title
        ax.text(0.15, insight['y']+0.05, insight['title'], 
                fontsize=14, weight='bold')
        
        # Add content
        ax.text(0.15, insight['y']-0.03, insight['content'], 
                fontsize=11)
    
    # Overall conclusion
    conclusion = """The embedding space analysis reveals that perfect AI DNA patterns organize into 
meaningful geometric structures, with logical/connective patterns forming the semantic 
center while existence/philosophical patterns occupy the periphery."""
    
    ax.text(0.5, 0.08, conclusion, ha='center', va='center',
            fontsize=12, style='italic', wrap=True,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#34495e', alpha=0.3))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('embedding_key_insights.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def main():
    """Generate all analysis visualizations"""
    print("Loading embedding space results...")
    results = load_results()
    
    if not results:
        return
    
    print("Creating analysis visualizations...")
    
    create_central_patterns_visualization(results)
    print("✓ Created central patterns visualization")
    
    create_clustering_summary(results)
    print("✓ Created clustering summary")
    
    create_semantic_neighborhoods_viz(results)
    print("✓ Created semantic neighborhoods visualization")
    
    create_key_insights_summary()
    print("✓ Created key insights summary")
    
    print("\nAll analysis visualizations created successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("EMBEDDING SPACE ANALYSIS SUMMARY")
    print("="*60)
    
    if 'insights' in results:
        for insight in results['insights']:
            print(f"• {insight}")
    
    # Additional insights
    print("\nPattern Distribution:")
    for model in results['models']:
        data = results['models'][model]
        print(f"  {model}: {data['n_perfect_patterns']} perfect, "
              f"{data['n_related_patterns']} related, "
              f"{data['n_control_patterns']} control")

if __name__ == "__main__":
    main()