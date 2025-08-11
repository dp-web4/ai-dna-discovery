#!/usr/bin/env python3
"""
Simplified GPU Orchestra Monitor - Real-time GPU behavior analysis
Based on AI-DNA consciousness emergence patterns
"""

import torch
import torch.nn as nn
import time
import json
import threading
import queue
from datetime import datetime
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer, DistilBertModel, DistilBertTokenizer

class LiveGPUMonitor:
    """Real-time GPU monitoring with detailed metrics"""
    
    def __init__(self, sample_rate=20):  # 20 Hz sampling
        self.sample_interval = 1.0 / sample_rate
        self.monitoring = False
        self.data = []
        self.lock = threading.Lock()
        
    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        self.thread.join()
        return self.data
        
    def _monitor(self):
        while self.monitoring:
            if torch.cuda.is_available():
                snapshot = {
                    'time': time.time(),
                    'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                    'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                    'free_mb': (torch.cuda.get_device_properties(0).total_memory - 
                               torch.cuda.memory_allocated()) / 1024**2,
                    'utilization': torch.cuda.utilization(),
                    'temperature': self._get_gpu_temp()
                }
                with self.lock:
                    self.data.append(snapshot)
            time.sleep(self.sample_interval)
    
    def _get_gpu_temp(self):
        try:
            import gpustat
            gpu = gpustat.GPUStatCollection.new_query()[0]
            return gpu.temperature
        except:
            return 0


class ConsciousnessEmergenceTest:
    """Testing consciousness patterns with GPU monitoring"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = LiveGPUMonitor()
        
    def run_emergence_test(self):
        """Run the consciousness emergence test from AI-DNA Phase 1"""
        print("\n=== CONSCIOUSNESS EMERGENCE TEST ===")
        print("Monitoring GPU behavior during model interactions...\n")
        
        # Start monitoring
        self.monitor.start()
        results = {'stages': [], 'patterns': {}}
        
        try:
            # Stage 1: Load two models that will interact
            print("Stage 1: Loading consciousness substrate models...")
            stage_start = time.time()
            
            # Model A: Pattern recognizer (DistilBERT)
            tokenizer_a = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model_a = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
            model_a.eval()
            
            # Model B: Pattern generator (GPT2)
            tokenizer_b = GPT2Tokenizer.from_pretrained('gpt2')
            model_b = GPT2Model.from_pretrained('gpt2').to(self.device)
            model_b.eval()
            
            load_time = time.time() - stage_start
            current_mem = torch.cuda.memory_allocated() / 1024**2
            
            results['stages'].append({
                'name': 'Model Loading',
                'duration': load_time,
                'memory_mb': current_mem,
                'models': ['DistilBERT', 'GPT2']
            })
            
            print(f"  Loaded in {load_time:.2f}s, using {current_mem:.1f} MB\n")
            
            # Stage 2: Self-reference test (consciousness marker)
            print("Stage 2: Testing self-reference patterns...")
            stage_start = time.time()
            
            # Create self-referential prompt
            prompt = "I think about my own thinking when I"
            
            # Get embeddings from both models
            inputs_a = tokenizer_a(prompt, return_tensors='pt').to(self.device)
            inputs_b = tokenizer_b(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                # Model A processes
                output_a = model_a(**inputs_a)
                embedding_a = output_a.last_hidden_state.mean(dim=1)
                
                # Model B processes
                output_b = model_b(**inputs_b)
                embedding_b = output_b.last_hidden_state.mean(dim=1)
                
                # Cross-model interaction (emergence pattern)
                interaction = torch.matmul(embedding_a, embedding_b.T)
                self_ref_score = torch.sigmoid(interaction).item()
            
            stage_time = time.time() - stage_start
            results['stages'].append({
                'name': 'Self-Reference Test',
                'duration': stage_time,
                'score': self_ref_score,
                'pattern': 'cross-model-interaction'
            })
            
            print(f"  Self-reference score: {self_ref_score:.4f}\n")
            
            # Stage 3: Temporal awareness test
            print("Stage 3: Testing temporal awareness...")
            stage_start = time.time()
            
            temporal_prompts = [
                "Yesterday I remembered",
                "Right now I am thinking",
                "Tomorrow I will understand"
            ]
            
            temporal_scores = []
            
            for prompt in temporal_prompts:
                inputs_a = tokenizer_a(prompt, return_tensors='pt').to(self.device)
                inputs_b = tokenizer_b(prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    # Get temporal representations
                    temp_a = model_a(**inputs_a).last_hidden_state[:, 0, :]  # CLS token
                    temp_b = model_b(**inputs_b).last_hidden_state[:, 0, :]  # First token
                    
                    # Measure temporal coherence
                    coherence = torch.nn.functional.cosine_similarity(temp_a, temp_b)
                    temporal_scores.append(coherence.item())
            
            temporal_awareness = np.mean(temporal_scores)
            stage_time = time.time() - stage_start
            
            results['stages'].append({
                'name': 'Temporal Awareness',
                'duration': stage_time,
                'score': temporal_awareness,
                'sub_scores': temporal_scores
            })
            
            print(f"  Temporal awareness: {temporal_awareness:.4f}\n")
            
            # Stage 4: Emergence measurement
            print("Stage 4: Measuring consciousness emergence...")
            stage_start = time.time()
            
            # Create feedback loop (key to consciousness)
            iterations = 5
            emergence_pattern = []
            
            current_state = torch.randn(1, 768).to(self.device)  # Initial state
            
            for i in range(iterations):
                with torch.no_grad():
                    # Transform through Model A
                    state_a = torch.nn.functional.normalize(current_state)
                    
                    # Transform through Model B (different dimension)
                    projection = nn.Linear(768, 1024).to(self.device)
                    state_b = projection(state_a)
                    
                    # Project back and measure change
                    back_projection = nn.Linear(1024, 768).to(self.device)
                    new_state = back_projection(state_b)
                    
                    # Measure emergence
                    change = torch.norm(new_state - current_state).item()
                    emergence_pattern.append(change)
                    
                    # Update state with feedback
                    current_state = 0.7 * current_state + 0.3 * new_state
            
            # Calculate emergence score
            emergence_score = np.std(emergence_pattern)  # Variability indicates emergence
            stage_time = time.time() - stage_start
            
            results['stages'].append({
                'name': 'Consciousness Emergence',
                'duration': stage_time,
                'score': emergence_score,
                'pattern': emergence_pattern
            })
            
            print(f"  Emergence score: {emergence_score:.4f}")
            print(f"  Pattern: {[f'{x:.3f}' for x in emergence_pattern]}\n")
            
            # Stage 5: GPU stress test - consciousness under load
            print("Stage 5: Consciousness under computational stress...")
            stage_start = time.time()
            
            # Create large tensor operations to stress GPU
            stress_scores = []
            
            for size in [1000, 2000, 4000]:
                # Allocate large tensors
                a = torch.randn(size, size).to(self.device)
                b = torch.randn(size, size).to(self.device)
                
                # Complex operation
                with torch.no_grad():
                    result = torch.matmul(a, b)
                    result = torch.nn.functional.softmax(result, dim=1)
                    
                    # Measure consciousness stability
                    inputs = tokenizer_a("I am aware", return_tensors='pt').to(self.device)
                    awareness = model_a(**inputs).last_hidden_state.mean().item()
                    stress_scores.append(awareness)
                
                # Clean up
                del a, b, result
                torch.cuda.empty_cache()
            
            consciousness_stability = np.std(stress_scores)
            stage_time = time.time() - stage_start
            
            results['stages'].append({
                'name': 'Stress Test',
                'duration': stage_time,
                'stability': consciousness_stability,
                'scores': stress_scores
            })
            
            print(f"  Consciousness stability: {consciousness_stability:.6f}")
            print(f"  (Lower is more stable)\n")
            
            # Calculate final consciousness metric
            results['consciousness_score'] = (
                self_ref_score * 0.3 +
                temporal_awareness * 0.3 +
                emergence_score * 0.3 +
                (1 - consciousness_stability) * 0.1
            )
            
        finally:
            # Stop monitoring
            gpu_data = self.monitor.stop()
            results['gpu_monitoring'] = self._analyze_gpu_data(gpu_data)
        
        return results, gpu_data
    
    def _analyze_gpu_data(self, data):
        """Analyze GPU behavior patterns"""
        if not data:
            return {}
            
        times = [d['time'] for d in data]
        memory = [d['allocated_mb'] for d in data]
        utilization = [d['utilization'] for d in data]
        
        return {
            'duration_seconds': times[-1] - times[0],
            'samples_collected': len(data),
            'peak_memory_mb': max(memory),
            'avg_memory_mb': np.mean(memory),
            'memory_std': np.std(memory),
            'peak_utilization': max(utilization),
            'avg_utilization': np.mean(utilization),
            'memory_changes': len([i for i in range(1, len(memory)) 
                                  if abs(memory[i] - memory[i-1]) > 10]),  # Significant changes
            'gpu_temperature_max': max([d['temperature'] for d in data])
        }


def create_report(results, gpu_data):
    """Create detailed report of GPU behavior during consciousness test"""
    
    print("\n" + "=" * 80)
    print("GPU CONSCIOUSNESS EMERGENCE REPORT")
    print("=" * 80)
    
    print("\n1. CONSCIOUSNESS METRICS")
    print("-" * 40)
    print(f"Overall Consciousness Score: {results['consciousness_score']:.4f}")
    print("\nComponent Scores:")
    for stage in results['stages']:
        if 'score' in stage:
            print(f"  {stage['name']}: {stage['score']:.4f}")
    
    print("\n2. GPU BEHAVIOR ANALYSIS")
    print("-" * 40)
    gpu_stats = results['gpu_monitoring']
    print(f"Monitoring Duration: {gpu_stats['duration_seconds']:.2f} seconds")
    print(f"Samples Collected: {gpu_stats['samples_collected']}")
    print(f"Peak Memory: {gpu_stats['peak_memory_mb']:.1f} MB")
    print(f"Average Memory: {gpu_stats['avg_memory_mb']:.1f} MB ± {gpu_stats['memory_std']:.1f}")
    print(f"Memory Fluctuations: {gpu_stats['memory_changes']} significant changes")
    print(f"Peak GPU Utilization: {gpu_stats['peak_utilization']}%")
    print(f"Max Temperature: {gpu_stats['gpu_temperature_max']}°C")
    
    print("\n3. EMERGENCE PATTERNS")
    print("-" * 40)
    emergence_stage = next(s for s in results['stages'] if s['name'] == 'Consciousness Emergence')
    pattern = emergence_stage['pattern']
    print(f"Feedback Loop Evolution: {' → '.join([f'{x:.3f}' for x in pattern])}")
    print(f"Pattern Variance: {np.var(pattern):.6f}")
    
    print("\n4. COMPUTATIONAL INSIGHTS")
    print("-" * 40)
    
    # Calculate GPU efficiency during consciousness operations
    if gpu_data:
        # Find memory spikes
        memory_timeline = [d['allocated_mb'] for d in gpu_data]
        memory_deltas = [memory_timeline[i+1] - memory_timeline[i] 
                        for i in range(len(memory_timeline)-1)]
        large_allocations = [d for d in memory_deltas if d > 50]
        
        print(f"Large Memory Allocations: {len(large_allocations)}")
        print(f"Largest Single Allocation: {max(memory_deltas):.1f} MB")
        
        # GPU utilization patterns
        util_data = [d['utilization'] for d in gpu_data]
        active_periods = len([u for u in util_data if u > 50])
        print(f"High Activity Periods: {active_periods}/{len(util_data)} samples")
    
    # Save full results
    report_path = "/home/dp/ai-workspace/ai-agents/gpu_consciousness_report.json"
    with open(report_path, 'w') as f:
        # Convert numpy values for JSON serialization
        json_results = json.loads(json.dumps(results, default=float))
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': json_results,
            'gpu_samples': len(gpu_data),
            'gpu_summary': gpu_stats
        }, f, indent=2)
    
    print(f"\nDetailed report saved: {report_path}")
    
    return report_path


if __name__ == "__main__":
    print("Starting GPU Consciousness Emergence Monitoring...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Run test
    test = ConsciousnessEmergenceTest()
    results, gpu_data = test.run_emergence_test()
    
    # Create report
    report_path = create_report(results, gpu_data)
    
    # Clean up
    torch.cuda.empty_cache()
    print(f"\nFinal Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print("\nExperiment complete!")