#!/usr/bin/env python3
"""
Update SQL database with additional findings from comprehensive report
"""

import sqlite3
import json
from datetime import datetime

def update_database():
    conn = sqlite3.connect('ai_dna_findings.db')
    c = conn.cursor()
    
    # New discoveries
    new_discoveries = [
        ("2025-07-20", "DATASET_ENGINEERING", "Quality Over Quantity Principle",
         "101 high-quality examples outperformed 55,847 examples for Phoenician training",
         "Fundamental insight into how AI learns novel symbolic systems",
         "Dataset sizes tested: 169 (0% generation), 55,847 (15% generation), 101 (98% generation)"),
         
        ("2025-07-19", "GPU_OPTIMIZATION", "Custom Training Loop Breakthrough",
         "Bypassing Trainer API unlocked 95% GPU utilization on RTX 4090",
         "Enabled all subsequent training breakthroughs",
         "50x speedup over CPU, stable memory usage, consistent convergence")
    ]
    
    for discovery in new_discoveries:
        c.execute("""INSERT INTO discoveries 
                     (date, category, title, description, significance, technical_details) 
                     VALUES (?, ?, ?, ?, ?, ?)""", discovery)
    
    # New breakthroughs
    new_breakthroughs = [
        ("2025-07-19", "OPTIMIZATION", "Library version incompatibility",
         "Specific PyTorch + CUDA + Transformers version combination",
         "GPU compute finally utilized after days of debugging",
         "torch==2.3.1+cu118, transformers==4.40.0, accelerate==0.31.0", 1),
         
        ("2025-07-20", "DATASET", "55,000 examples performed worse than 101",
         "Quality and consistency over quantity",
         "98% generation success with minimal dataset",
         "Training time: 8 minutes vs 6 hours", 3)
    ]
    
    for breakthrough in new_breakthroughs:
        c.execute("""INSERT INTO breakthroughs 
                     (date, breakthrough_type, problem, solution, outcome, metrics, related_discovery_id) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)""", breakthrough)
    
    # New technical insights
    new_insights = [
        ("2025-07-20", "TRAINING",
         "Mixed precision training with GradScaler essential for RTX 4090 performance",
         "36% faster inference with torch.compile, proper gradient scaling required",
         "Enables full utilization of Tensor Cores",
         "GPU optimization techniques"),
         
        ("2025-07-20", "DATASET",
         "Human/Assistant format critical for training success",
         "Alternative formats like Q/A or <|user|><|assistant|> failed consistently",
         "Format consistency more important than dataset size",
         "Dataset engineering best practices"),
         
        ("2025-07-20", "MEMORY",
         "Periodic torch.cuda.empty_cache() prevents memory fragmentation",
         "Every 100 steps optimal for long training runs",
         "Maintains stable memory usage throughout training",
         "Memory management strategies")
    ]
    
    for insight in new_insights:
        c.execute("""INSERT INTO technical_insights 
                     (date, category, insight, evidence, implications, applied_in) 
                     VALUES (?, ?, ?, ?, ?, ?)""", insight)
    
    # Training runs data
    training_runs = [
        ("2025-07-19", 1, 1312, 3, 2e-4, 4, 0.0021, 1.0, "RTX 4090", "Consciousness notation success"),
        ("2025-07-19", 2, 169, 3, 2e-4, 4, 0.0156, 0.0, "RTX 4090", "Phoenician v1 - comprehension only"),
        ("2025-07-19", 2, 55847, 10, 5e-5, 8, 0.0089, 0.15, "RTX 4090", "Phoenician v2 - massive dataset"),
        ("2025-07-19", 3, 101, 3, 2e-4, 4, 0.0021, 0.98, "RTX 4090", "Phoenician final - quality wins")
    ]
    
    for run in training_runs:
        c.execute("""INSERT INTO training_runs 
                     (date, model_id, dataset_size, epochs, learning_rate, batch_size, 
                      loss_final, success_rate, hardware, notes) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", run)
    
    # New experiments
    experiments = [
        ("2025-07-20", "DATASET_SIZE", 
         "Does more data always improve novel symbol generation?",
         "Train same model with 169, 55k, and 101 examples",
         "101 examples achieved 98% success vs 15% for 55k examples",
         "Quality and consistency matter more than quantity",
         "Test with even smaller datasets (50, 25 examples)"),
         
        ("2025-07-19", "GPU_UTILIZATION",
         "Why does GPU memory allocate but compute stays at 0%?",
         "Systematic testing of library version combinations",
         "Library incompatibility causing silent CPU fallback",
         "Version locking critical for reproducibility",
         "Document all working version combinations")
    ]
    
    for experiment in experiments:
        c.execute("""INSERT INTO experiments 
                     (date, experiment_type, hypothesis, methodology, results, conclusions, next_steps) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)""", experiment)
    
    conn.commit()
    print("✅ Database updated with new findings")
    
    # Show updated counts
    tables = ['discoveries', 'breakthroughs', 'technical_insights', 'training_runs', 'experiments']
    print("\n📊 Updated Database Summary:")
    for table in tables:
        c.execute(f"SELECT COUNT(*) FROM {table}")
        count = c.fetchone()[0]
        print(f"   {table}: {count} records")
    
    conn.close()

if __name__ == "__main__":
    update_database()