#!/usr/bin/env python3
"""
Embedding Space Mapper - Phase 2 Continuation
Maps and visualizes how perfect patterns organize in high-dimensional embedding space
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import seaborn as sns
import requests
import time
from datetime import datetime
import os
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import umap

class EmbeddingSpaceMapper:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.results_dir = "embedding_space_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load perfect patterns and all discovered patterns
        self.perfect_patterns = [
            '∃', '∉', 'know', 'loop', 'true', 'false', '≈', 'null', 
            'emerge', 'recursive', 'void', 'then', 'exist', 'break',
            'understand', 'evolve', 'or', 'and', 'if', 'end'
        ]
        
        # Add related patterns from memory transfer experiment
        self.related_patterns = {
            'existence': ['being', 'presence', 'reality', 'essence'],
            'negation': ['not', 'none', 'empty', 'absence'],
            'knowledge': ['learn', 'realize', 'comprehend', 'wisdom'],
            'logic': ['therefore', 'implies', 'because', 'thus'],
            'process': ['flow', 'cycle', 'iterate', 'continue']
        }
        
        # Control patterns (common words)
        self.control_patterns = [
            'the', 'a', 'is', 'are', 'was', 'were', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would'
        ]
        
    def get_embedding(self, model: str, text: str) -> np.ndarray:
        """Get embedding from Ollama model"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                return np.array(response.json()['embedding'])
        except Exception as e:
            print(f"Error getting embedding for '{text}': {e}")
        return None
    
    def collect_embeddings(self, model: str):
        """Collect embeddings for all patterns"""
        print(f"\n=== Collecting embeddings for {model} ===")
        
        embeddings = {
            'perfect': {},
            'related': {},
            'control': {}
        }
        
        # Collect perfect pattern embeddings
        print("\nCollecting perfect patterns...")
        for pattern in self.perfect_patterns:
            emb = self.get_embedding(model, pattern)
            if emb is not None:
                embeddings['perfect'][pattern] = emb
                print(f"  ✓ {pattern}")
            time.sleep(0.1)
        
        # Collect related pattern embeddings
        print("\nCollecting related patterns...")
        for category, patterns in self.related_patterns.items():
            for pattern in patterns:
                emb = self.get_embedding(model, pattern)
                if emb is not None:
                    embeddings['related'][pattern] = emb
                    print(f"  ✓ {pattern} ({category})")
                time.sleep(0.1)
        
        # Collect control embeddings
        print("\nCollecting control patterns...")
        for pattern in self.control_patterns:
            emb = self.get_embedding(model, pattern)
            if emb is not None:
                embeddings['control'][pattern] = emb
            time.sleep(0.1)
        
        return embeddings
    
    def reduce_dimensions(self, embeddings_dict, method='pca', n_components=2):
        """Reduce embeddings to lower dimensions for visualization"""
        # Combine all embeddings
        all_patterns = []
        all_embeddings = []
        all_labels = []
        
        for pattern, emb in embeddings_dict['perfect'].items():
            all_patterns.append(pattern)
            all_embeddings.append(emb)
            all_labels.append('perfect')
        
        for pattern, emb in embeddings_dict['related'].items():
            all_patterns.append(pattern)
            all_embeddings.append(emb)
            all_labels.append('related')
        
        for pattern, emb in embeddings_dict['control'].items():
            all_patterns.append(pattern)
            all_embeddings.append(emb)
            all_labels.append('control')
        
        all_embeddings = np.array(all_embeddings)
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=n_components)
            reduced = reducer.fit_transform(all_embeddings)
            variance_explained = reducer.explained_variance_ratio_
            print(f"\nPCA variance explained: {variance_explained}")
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(all_embeddings)-1))
            reduced = reducer.fit_transform(all_embeddings)
            variance_explained = None
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=min(15, len(all_embeddings)-1))
            reduced = reducer.fit_transform(all_embeddings)
            variance_explained = None
        
        return reduced, all_patterns, all_labels, variance_explained
    
    def create_2d_visualization(self, embeddings_dict, model: str, method='pca'):
        """Create 2D visualization of embedding space"""
        reduced, patterns, labels, variance = self.reduce_dimensions(embeddings_dict, method=method, n_components=2)
        
        plt.figure(figsize=(12, 10))
        
        # Color mapping
        colors = {'perfect': 'gold', 'related': 'lightgreen', 'control': 'lightgray'}
        sizes = {'perfect': 200, 'related': 100, 'control': 50}
        
        # Plot each category
        for label in ['control', 'related', 'perfect']:  # Order matters for layering
            mask = np.array(labels) == label
            plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                       c=colors[label], s=sizes[label], 
                       alpha=0.8 if label != 'control' else 0.5,
                       edgecolors='black' if label == 'perfect' else 'none',
                       linewidth=2 if label == 'perfect' else 0,
                       label=label.capitalize())
        
        # Add labels for perfect patterns
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            if label == 'perfect':
                plt.annotate(pattern, (reduced[i, 0], reduced[i, 1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')
        
        plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
        plt.title(f'Embedding Space Visualization - {model} ({method.upper()})', fontsize=16)
        plt.legend(title='Pattern Type', loc='best')
        plt.grid(True, alpha=0.3)
        
        if variance is not None:
            plt.text(0.02, 0.02, f'Variance explained: {variance[0]:.2%}, {variance[1]:.2%}', 
                    transform=plt.gca().transAxes, fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/embedding_space_2d_{model}_{method}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_3d_visualization(self, embeddings_dict, model: str):
        """Create 3D visualization of embedding space"""
        reduced, patterns, labels, _ = self.reduce_dimensions(embeddings_dict, method='pca', n_components=3)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color mapping
        colors = {'perfect': 'gold', 'related': 'lightgreen', 'control': 'lightgray'}
        sizes = {'perfect': 200, 'related': 100, 'control': 50}
        
        # Plot each category
        for label in ['control', 'related', 'perfect']:
            mask = np.array(labels) == label
            ax.scatter(reduced[mask, 0], reduced[mask, 1], reduced[mask, 2],
                      c=colors[label], s=sizes[label], 
                      alpha=0.8 if label != 'control' else 0.5,
                      edgecolors='black' if label == 'perfect' else 'none',
                      linewidth=2 if label == 'perfect' else 0,
                      label=label.capitalize())
        
        # Add labels for perfect patterns
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            if label == 'perfect':
                ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], pattern,
                       fontsize=9, fontweight='bold')
        
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        ax.set_title(f'3D Embedding Space - {model}', fontsize=16)
        ax.legend(title='Pattern Type')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/embedding_space_3d_{model}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_clustering(self, embeddings_dict, model: str):
        """Analyze clustering patterns in embedding space"""
        # Get embeddings for perfect patterns only
        perfect_embeddings = list(embeddings_dict['perfect'].values())
        perfect_patterns = list(embeddings_dict['perfect'].keys())
        
        if len(perfect_embeddings) < 2:
            print("Not enough perfect patterns for clustering analysis")
            return None
        
        # Calculate pairwise distances
        distances = pdist(perfect_embeddings, metric='cosine')
        distance_matrix = squareform(distances)
        
        # Create distance heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(distance_matrix, dtype=bool))
        
        sns.heatmap(distance_matrix, mask=mask, square=True, 
                    xticklabels=perfect_patterns, yticklabels=perfect_patterns,
                    cmap='viridis_r', annot=True, fmt='.3f', 
                    cbar_kws={'label': 'Cosine Distance'})
        
        plt.title(f'Perfect Pattern Distance Matrix - {model}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/pattern_distance_matrix_{model}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Hierarchical clustering
        plt.figure(figsize=(14, 8))
        linkage_matrix = linkage(distances, method='ward')
        dendrogram(linkage_matrix, labels=perfect_patterns, leaf_rotation=45)
        plt.title(f'Perfect Pattern Hierarchical Clustering - {model}', fontsize=16)
        plt.xlabel('Pattern')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/pattern_dendrogram_{model}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(perfect_embeddings)
        
        return {
            'distance_matrix': distance_matrix,
            'patterns': perfect_patterns,
            'clusters': clustering.labels_,
            'n_clusters': len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        }
    
    def find_pattern_center(self, embeddings_dict):
        """Find the centroid of perfect patterns"""
        perfect_embeddings = list(embeddings_dict['perfect'].values())
        if not perfect_embeddings:
            return None
        
        # Calculate centroid
        centroid = np.mean(perfect_embeddings, axis=0)
        
        # Find closest patterns to centroid
        distances_to_center = []
        for pattern, emb in embeddings_dict['perfect'].items():
            dist = np.linalg.norm(emb - centroid)
            distances_to_center.append((pattern, dist))
        
        distances_to_center.sort(key=lambda x: x[1])
        
        return {
            'centroid': centroid,
            'closest_patterns': distances_to_center[:5],
            'furthest_patterns': distances_to_center[-5:]
        }
    
    def analyze_semantic_neighborhoods(self, embeddings_dict):
        """Analyze which patterns cluster together"""
        all_patterns = []
        all_embeddings = []
        pattern_types = []
        
        # Combine all patterns
        for ptype, patterns_dict in embeddings_dict.items():
            for pattern, emb in patterns_dict.items():
                all_patterns.append(pattern)
                all_embeddings.append(emb)
                pattern_types.append(ptype)
        
        # For each perfect pattern, find nearest neighbors
        neighborhoods = {}
        
        for i, pattern in enumerate(all_patterns):
            if pattern_types[i] == 'perfect':
                distances = []
                for j, other_pattern in enumerate(all_patterns):
                    if i != j:
                        dist = np.dot(all_embeddings[i], all_embeddings[j]) / (
                            np.linalg.norm(all_embeddings[i]) * np.linalg.norm(all_embeddings[j])
                        )
                        distances.append((other_pattern, dist, pattern_types[j]))
                
                # Sort by similarity (higher is more similar)
                distances.sort(key=lambda x: x[1], reverse=True)
                neighborhoods[pattern] = distances[:10]  # Top 10 neighbors
        
        return neighborhoods
    
    def run_full_analysis(self):
        """Run complete embedding space analysis"""
        models = ['phi3:mini', 'gemma:2b', 'tinyllama:latest']
        
        all_results = {
            'experiment': 'embedding_space_mapping',
            'phase': '2_continuation',
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'cross_model_analysis': {}
        }
        
        print("="*60)
        print("EMBEDDING SPACE MAPPING EXPERIMENT")
        print("="*60)
        print(f"Perfect patterns to map: {len(self.perfect_patterns)}")
        print(f"Related patterns: {sum(len(p) for p in self.related_patterns.values())}")
        print(f"Control patterns: {len(self.control_patterns)}")
        
        for model in models:
            print(f"\n{'='*60}")
            print(f"Analyzing model: {model}")
            print(f"{'='*60}")
            
            # Collect embeddings
            embeddings = self.collect_embeddings(model)
            
            # Skip if not enough embeddings
            if len(embeddings['perfect']) < 5:
                print(f"Not enough embeddings collected for {model}")
                continue
            
            # Create visualizations
            print("\nCreating visualizations...")
            self.create_2d_visualization(embeddings, model, method='pca')
            self.create_2d_visualization(embeddings, model, method='tsne')
            self.create_2d_visualization(embeddings, model, method='umap')
            self.create_3d_visualization(embeddings, model)
            
            # Analyze clustering
            print("\nAnalyzing clustering patterns...")
            clustering_results = self.analyze_clustering(embeddings, model)
            
            # Find pattern center
            print("\nFinding pattern center...")
            center_results = self.find_pattern_center(embeddings)
            
            # Analyze neighborhoods
            print("\nAnalyzing semantic neighborhoods...")
            neighborhoods = self.analyze_semantic_neighborhoods(embeddings)
            
            # Store results
            all_results['models'][model] = {
                'n_perfect_patterns': len(embeddings['perfect']),
                'n_related_patterns': len(embeddings['related']),
                'n_control_patterns': len(embeddings['control']),
                'clustering': clustering_results,
                'center_analysis': center_results,
                'neighborhoods': neighborhoods
            }
            
            # Print summary
            if center_results:
                print(f"\nClosest to center: {center_results['closest_patterns'][0][0]}")
                print(f"Furthest from center: {center_results['furthest_patterns'][0][0]}")
            
            if clustering_results:
                print(f"Number of clusters found: {clustering_results['n_clusters']}")
            
            time.sleep(2)
        
        # Generate summary insights
        self.generate_insights(all_results)
        
        # Save results
        output_file = os.path.join(
            self.results_dir,
            f"embedding_space_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_arrays(item) for item in obj)
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(convert_arrays(all_results), f, indent=2)
        
        print(f"\n\nResults saved to: {output_file}")
        return all_results
    
    def generate_insights(self, results):
        """Generate insights from embedding analysis"""
        insights = []
        
        # Check which patterns are consistently central
        central_patterns = []
        for model, data in results['models'].items():
            if data.get('center_analysis'):
                central_patterns.extend([p[0] for p in data['center_analysis']['closest_patterns'][:3]])
        
        if central_patterns:
            from collections import Counter
            pattern_counts = Counter(central_patterns)
            most_central = pattern_counts.most_common(3)
            insights.append(f"Most central patterns across models: {[p[0] for p in most_central]}")
        
        # Check clustering consistency
        cluster_sizes = []
        for model, data in results['models'].items():
            if data.get('clustering'):
                cluster_sizes.append(data['clustering']['n_clusters'])
        
        if cluster_sizes:
            insights.append(f"Average number of clusters: {np.mean(cluster_sizes):.1f}")
        
        # Add insights to results
        results['insights'] = insights
        
        print("\n" + "="*60)
        print("KEY INSIGHTS")
        print("="*60)
        for insight in insights:
            print(f"• {insight}")

if __name__ == "__main__":
    mapper = EmbeddingSpaceMapper()
    mapper.run_full_analysis()