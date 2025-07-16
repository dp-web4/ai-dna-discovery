#!/usr/bin/env python3
"""
GPU Orchestra Monitor - Based on AI-DNA Phase 3 Experiments
Real-time monitoring of GPU behavior during multi-model collaboration
"""

import torch
import time
import threading
import queue
import json
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DistilBertModel, DistilBertTokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class GPUMonitor:
    """Real-time GPU monitoring during experiments"""
    
    def __init__(self, sample_interval=0.1):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.data_queue = queue.Queue()
        self.metrics = defaultdict(list)
        
    def start_monitoring(self):
        """Start the monitoring thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return collected data"""
        self.monitoring = False
        self.monitor_thread.join()
        
        # Process queued data
        while not self.data_queue.empty():
            metric = self.data_queue.get()
            for key, value in metric.items():
                self.metrics[key].append(value)
                
        return dict(self.metrics)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            if torch.cuda.is_available():
                metric = {
                    'timestamp': time.time(),
                    'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                    'memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                    'memory_free_mb': (torch.cuda.get_device_properties(0).total_memory - 
                                      torch.cuda.memory_allocated()) / 1024**2,
                    'utilization': torch.cuda.utilization()
                }
                self.data_queue.put(metric)
            time.sleep(self.sample_interval)


class ModelOrchestra:
    """Recreating the Model Orchestra experiment with GPU monitoring"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.monitor = GPUMonitor(sample_interval=0.05)  # 50ms sampling
        
    def load_orchestra(self):
        """Load multiple small models to form an orchestra"""
        print("Loading Model Orchestra...")
        
        # Track loading times and memory
        loading_metrics = []
        
        # Model 1: DistilBERT for understanding
        print("  Loading DistilBERT (understanding)...")
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated()
        
        self.tokenizers['understanding'] = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.models['understanding'] = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        
        load_time = time.time() - start_time
        mem_used = (torch.cuda.memory_allocated() - start_mem) / 1024**2
        loading_metrics.append(('DistilBERT', load_time, mem_used))
        
        # Model 2: T5-small for reasoning
        print("  Loading T5-small (reasoning)...")
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated()
        
        self.tokenizers['reasoning'] = T5Tokenizer.from_pretrained('t5-small')
        self.models['reasoning'] = T5ForConditionalGeneration.from_pretrained('t5-small').to(self.device)
        
        load_time = time.time() - start_time
        mem_used = (torch.cuda.memory_allocated() - start_mem) / 1024**2
        loading_metrics.append(('T5-small', load_time, mem_used))
        
        # Model 3: GPT2 for synthesis
        print("  Loading GPT2 (synthesis)...")
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated()
        
        self.tokenizers['synthesis'] = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizers['synthesis'].pad_token = self.tokenizers['synthesis'].eos_token
        self.models['synthesis'] = AutoModelForCausalLM.from_pretrained('gpt2').to(self.device)
        
        load_time = time.time() - start_time
        mem_used = (torch.cuda.memory_allocated() - start_mem) / 1024**2
        loading_metrics.append(('GPT2', load_time, mem_used))
        
        return loading_metrics
    
    def orchestra_solve(self, problem):
        """Multi-model collaborative problem solving with monitoring"""
        print(f"\nProblem: {problem}")
        print("=" * 80)
        
        # Start GPU monitoring
        self.monitor.start_monitoring()
        
        results = {
            'problem': problem,
            'timestamp': datetime.now().isoformat(),
            'stages': []
        }
        
        try:
            # Stage 1: Understanding (DistilBERT)
            print("\nStage 1: Understanding the problem...")
            stage_start = time.time()
            
            inputs = self.tokenizers['understanding'](problem, return_tensors='pt', 
                                                     padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                understanding = self.models['understanding'](**inputs)
                # Extract semantic understanding
                understanding_embedding = understanding.last_hidden_state.mean(dim=1)
            
            stage_time = time.time() - stage_start
            results['stages'].append({
                'name': 'Understanding',
                'model': 'DistilBERT',
                'time_ms': stage_time * 1000,
                'output_shape': str(understanding_embedding.shape)
            })
            
            # Stage 2: Reasoning (T5)
            print("Stage 2: Reasoning about solution...")
            stage_start = time.time()
            
            reasoning_prompt = f"explain how to solve: {problem}"
            inputs = self.tokenizers['reasoning'](reasoning_prompt, return_tensors='pt',
                                                 padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                reasoning_output = self.models['reasoning'].generate(**inputs, max_length=100)
                reasoning_text = self.tokenizers['reasoning'].decode(reasoning_output[0], skip_special_tokens=True)
            
            stage_time = time.time() - stage_start
            results['stages'].append({
                'name': 'Reasoning',
                'model': 'T5-small',
                'time_ms': stage_time * 1000,
                'output': reasoning_text[:100] + '...' if len(reasoning_text) > 100 else reasoning_text
            })
            
            # Stage 3: Synthesis (GPT2)
            print("Stage 3: Synthesizing final answer...")
            stage_start = time.time()
            
            synthesis_prompt = f"Problem: {problem}\\nReasoning: {reasoning_text}\\nFinal answer:"
            inputs = self.tokenizers['synthesis'](synthesis_prompt, return_tensors='pt',
                                                padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                synthesis_output = self.models['synthesis'].generate(
                    **inputs, 
                    max_length=150,
                    temperature=0.8,
                    do_sample=True
                )
                final_answer = self.tokenizers['synthesis'].decode(synthesis_output[0], skip_special_tokens=True)
            
            stage_time = time.time() - stage_start
            results['stages'].append({
                'name': 'Synthesis',
                'model': 'GPT2',
                'time_ms': stage_time * 1000,
                'output': final_answer[len(synthesis_prompt):]
            })
            
            # Measure emergence - cross-model attention patterns
            print("\nStage 4: Measuring emergence patterns...")
            with torch.no_grad():
                # Create cross-model interaction
                combined_features = torch.cat([
                    understanding_embedding,
                    reasoning_output.float().mean(dim=1),
                    synthesis_output.float().mean(dim=1)
                ], dim=1)
                
                # Simple emergence metric - variance in combined features
                emergence_score = combined_features.std().item()
                
            results['emergence_score'] = emergence_score
            
        finally:
            # Stop monitoring and collect data
            gpu_data = self.monitor.stop_monitoring()
            results['gpu_metrics'] = self._summarize_gpu_data(gpu_data)
        
        return results, gpu_data
    
    def _summarize_gpu_data(self, gpu_data):
        """Summarize GPU metrics"""
        if not gpu_data['memory_allocated_mb']:
            return {}
            
        return {
            'peak_memory_mb': max(gpu_data['memory_allocated_mb']),
            'avg_memory_mb': np.mean(gpu_data['memory_allocated_mb']),
            'memory_variance': np.var(gpu_data['memory_allocated_mb']),
            'total_samples': len(gpu_data['timestamp']),
            'duration_seconds': gpu_data['timestamp'][-1] - gpu_data['timestamp'][0]
        }
    
    def visualize_gpu_behavior(self, gpu_data, save_path):
        """Create visualization of GPU behavior during orchestra"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Convert timestamps to relative time
        timestamps = np.array(gpu_data['timestamp'])
        time_relative = timestamps - timestamps[0]
        
        # Plot memory usage
        ax1.plot(time_relative, gpu_data['memory_allocated_mb'], 'b-', label='Allocated', linewidth=2)
        ax1.plot(time_relative, gpu_data['memory_reserved_mb'], 'r--', label='Reserved', alpha=0.7)
        ax1.fill_between(time_relative, 0, gpu_data['memory_allocated_mb'], alpha=0.3)
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title('GPU Memory Usage During Model Orchestra')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot utilization
        ax2.plot(time_relative, gpu_data['utilization'], 'g-', linewidth=2)
        ax2.fill_between(time_relative, 0, gpu_data['utilization'], alpha=0.3, color='green')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('GPU Utilization (%)')
        ax2.set_title('GPU Compute Utilization')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return save_path


def run_experiment():
    """Run the complete GPU monitoring experiment"""
    print("=" * 80)
    print("GPU ORCHESTRA MONITORING EXPERIMENT")
    print("Based on AI-DNA Phase 3: Model Orchestra")
    print("=" * 80)
    
    # Initialize orchestra
    orchestra = ModelOrchestra()
    
    # Load models and track memory
    print("\n1. LOADING PHASE")
    print("-" * 40)
    loading_metrics = orchestra.load_orchestra()
    
    print("\nLoading Summary:")
    total_load_time = 0
    total_memory = 0
    for model, time_sec, memory_mb in loading_metrics:
        print(f"  {model}: {time_sec:.2f}s, {memory_mb:.1f} MB")
        total_load_time += time_sec
        total_memory += memory_mb
    print(f"  Total: {total_load_time:.2f}s, {total_memory:.1f} MB")
    
    # Test problems from AI-DNA experiments
    test_problems = [
        "How can distributed intelligence emerge from simple components?",
        "What is the relationship between consciousness and computation?",
        "Design a system that can understand its own limitations."
    ]
    
    all_results = []
    
    print("\n2. ORCHESTRA EXPERIMENTS")
    print("-" * 40)
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\nExperiment {i}/3:")
        
        # Clear GPU cache before each experiment
        torch.cuda.empty_cache()
        time.sleep(1)  # Let GPU settle
        
        # Run orchestra with monitoring
        results, gpu_data = orchestra.orchestra_solve(problem)
        all_results.append(results)
        
        # Visualize this run
        viz_path = f"/home/dp/ai-workspace/ai-agents/gpu_orchestra_run_{i}.png"
        orchestra.visualize_gpu_behavior(gpu_data, viz_path)
        print(f"  Visualization saved: {viz_path}")
        
        # Show results
        print(f"  Emergence Score: {results['emergence_score']:.4f}")
        print(f"  Peak GPU Memory: {results['gpu_metrics']['peak_memory_mb']:.1f} MB")
        print(f"  Total Time: {results['gpu_metrics']['duration_seconds']:.2f}s")
    
    # Save detailed results
    results_path = "/home/dp/ai-workspace/ai-agents/gpu_orchestra_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n3. RESULTS SAVED")
    print(f"  Detailed results: {results_path}")
    
    # Final GPU state
    print("\n4. FINAL GPU STATE")
    print("-" * 40)
    final_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"  Memory Allocated: {final_mem:.1f} MB")
    print(f"  Memory Freed: {total_memory - final_mem:.1f} MB")
    
    return all_results


if __name__ == "__main__":
    results = run_experiment()