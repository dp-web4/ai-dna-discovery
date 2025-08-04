# HRM-CPTE Implementation Roadmap

## Quick Start Guide

```bash
# Set up awareness development environment
cd awareness/
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run initial tests
python test_sensor_fusion.py
python test_memory_integration.py
python test_cpte_routing.py
```

## Directory Structure

```
awareness/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── core/
│   ├── __init__.py
│   ├── optimized_hrm.py        # Main HRM with all integrations
│   ├── sensor_fusion.py        # Trust-weighted sensor integration
│   ├── temporal_memory.py      # Hierarchical memory system
│   └── cpte_router.py          # CPTE routing logic
├── data/
│   ├── generators/
│   │   ├── awareness_scenarios.py
│   │   ├── sensor_simulator.py
│   │   └── knowledge_gaps.py
│   └── datasets/               # Generated training data
├── models/
│   ├── checkpoints/            # Saved model states
│   └── configs/                # Model configurations
├── tests/
│   ├── test_integration.py     # Full system tests
│   ├── test_edge_deployment.py # Jetson-specific tests
│   └── benchmarks/             # Performance benchmarks
└── experiments/
    ├── ablations/              # Component ablation studies
    ├── distributed/            # Multi-device awareness
    └── notebooks/              # Jupyter experiments
```

## Code Templates

### 1. Sensor Fusion Module
```python
# awareness/core/sensor_fusion.py
import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class SensorReading:
    """Unified sensor reading format"""
    sensor_type: str
    data: torch.Tensor
    confidence: float
    timestamp: float

class TrustWeightedFusion(nn.Module):
    """Trust-weighted multi-modal sensor fusion"""
    
    def __init__(self, sensor_configs: Dict[str, int]):
        super().__init__()
        self.sensor_encoders = nn.ModuleDict()
        self.trust_weights = nn.ParameterDict()
        
        total_dim = 0
        for sensor_type, input_dim in sensor_configs.items():
            # Sensor-specific encoders
            self.sensor_encoders[sensor_type] = self._build_encoder(
                sensor_type, input_dim
            )
            # Learnable trust weights
            self.trust_weights[sensor_type] = nn.Parameter(
                torch.ones(1) * 0.5
            )
            total_dim += self._get_encoding_dim(sensor_type)
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
    def forward(self, sensor_data: Dict[str, SensorReading]) -> Tuple[torch.Tensor, Dict]:
        encodings = []
        trust_scores = {}
        
        for sensor_type, reading in sensor_data.items():
            if sensor_type in self.sensor_encoders:
                # Encode sensor data
                encoding = self.sensor_encoders[sensor_type](reading.data)
                
                # Apply trust weighting
                trust = torch.sigmoid(self.trust_weights[sensor_type])
                weighted_encoding = encoding * trust * reading.confidence
                
                encodings.append(weighted_encoding)
                trust_scores[sensor_type] = trust.item()
                
        # Fuse all sensors
        fused = torch.cat(encodings, dim=-1)
        awareness_vector = self.fusion_layer(fused)
        
        return awareness_vector, trust_scores
```

### 2. Temporal Memory Integration
```python
# awareness/core/temporal_memory.py
class HierarchicalMemory(nn.Module):
    """Memory system with multiple timescales"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Sensory buffer (100ms window)
        self.sensory_buffer = nn.LSTM(
            input_dim, 64, batch_first=True
        )
        
        # Working memory (10s window)
        self.working_memory = nn.LSTM(
            64, 128, num_layers=2, batch_first=True
        )
        
        # Episodic memory (minutes)
        self.episodic_memory = MemoryBank(
            size=1000, dim=128, 
            decay_rate=0.99
        )
        
        # Semantic markers (permanent)
        self.semantic_markers = CPTEMarkerBank(
            size=10000, dim=32
        )
        
        # Memory gate for integration
        self.memory_gate = nn.Sequential(
            nn.Linear(352, 256),  # All memory dims
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
```

### 3. CPTE Router
```python
# awareness/core/cpte_router.py
class CPTERouter(nn.Module):
    """Routes queries to appropriate CPTEs based on confidence"""
    
    def __init__(self, hidden_dim: int, num_domains: int):
        super().__init__()
        
        # Confidence assessment network
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains),
            nn.Softmax(dim=-1)
        )
        
        # Routing threshold (learnable)
        self.threshold = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, query_state: torch.Tensor) -> Dict:
        # Assess confidence in internal knowledge
        confidence = self.confidence_net(query_state)
        
        # Classify domain if routing needed
        domain_probs = self.domain_classifier(query_state)
        
        # Routing decision
        route_external = confidence < torch.sigmoid(self.threshold)
        
        return {
            'confidence': confidence,
            'route_external': route_external,
            'domain_probs': domain_probs,
            'selected_domain': torch.argmax(domain_probs, dim=-1)
        }
```

## Training Data Specification

### Scenario Generator
```python
# awareness/data/generators/awareness_scenarios.py
class AwarenessScenarioGenerator:
    """Generates integrated awareness training scenarios"""
    
    def __init__(self):
        self.env_types = ['indoor', 'outdoor', 'urban', 'nature']
        self.task_types = ['navigation', 'identification', 'planning', 'diagnosis']
        self.knowledge_domains = ['physics', 'biology', 'engineering', 'social']
        
    def generate_scenario(self, difficulty: float = 0.5):
        # 1. Environment setup
        env = self.create_environment(
            env_type=random.choice(self.env_types),
            complexity=difficulty
        )
        
        # 2. Sensor configuration
        sensors = self.configure_sensors(env, noise_level=difficulty * 0.5)
        
        # 3. Task generation
        task = self.create_task(
            task_type=random.choice(self.task_types),
            env=env,
            difficulty=difficulty
        )
        
        # 4. Knowledge requirements
        knowledge_gaps = self.identify_knowledge_requirements(task, env)
        
        # 5. Temporal sequence
        sequence = self.generate_temporal_sequence(
            env, task, 
            length=int(10 + difficulty * 50)
        )
        
        return {
            'environment': env,
            'sensors': sensors,
            'task': task,
            'knowledge_gaps': knowledge_gaps,
            'temporal_sequence': sequence,
            'ground_truth': self.solve_optimally(env, task)
        }
```

## Evaluation Benchmarks

### 1. Situational Awareness Benchmark (SAB)
```python
# awareness/tests/benchmarks/situational_awareness.py
class SituationalAwarenessBenchmark:
    """Comprehensive awareness evaluation"""
    
    def __init__(self):
        self.scenarios = self.load_test_scenarios()
        self.metrics = {
            'location_accuracy': [],
            'context_understanding': [],
            'decision_quality': [],
            'knowledge_efficiency': [],
            'temporal_consistency': []
        }
        
    def evaluate_model(self, model, num_scenarios=100):
        for scenario in self.scenarios[:num_scenarios]:
            # Run model
            predictions = model.process_scenario(scenario)
            
            # Evaluate each aspect
            self.metrics['location_accuracy'].append(
                self.evaluate_location(predictions, scenario)
            )
            self.metrics['context_understanding'].append(
                self.evaluate_context(predictions, scenario)
            )
            # ... etc
            
        return self.compute_aggregate_metrics()
```

### 2. Edge Deployment Benchmark
```python
# awareness/tests/benchmarks/edge_performance.py
class EdgePerformanceBenchmark:
    """Jetson-specific performance testing"""
    
    def profile_model(self, model):
        # Memory usage
        memory_stats = self.measure_memory_usage(model)
        
        # Inference speed
        latency_stats = self.measure_latency(model, num_runs=1000)
        
        # Power consumption (if on Jetson)
        power_stats = self.measure_power_consumption(model)
        
        # Accuracy under constraints
        accuracy_degraded = self.test_quantization_impact(model)
        
        return {
            'memory_mb': memory_stats['peak_mb'],
            'latency_ms': latency_stats['p95'],
            'power_watts': power_stats['average'],
            'accuracy_loss': accuracy_degraded
        }
```

## Integration with Existing Systems

### 1. Memory System Bridge
```python
# awareness/core/memory_bridge.py
from memory.enhanced_memory_system import HierarchicalMemory as MemoryDB

class MemorySystemBridge:
    """Connects awareness to existing memory database"""
    
    def __init__(self, db_path: str):
        self.memory_db = MemoryDB(db_path)
        
    def inject_awareness_context(self, awareness_state):
        """Store awareness state in memory"""
        self.memory_db.store_with_confidence(
            content=f"Awareness state: {awareness_state}",
            memory_type='episodic',
            session_id=f"awareness_{time.time()}",
            source_confidence=awareness_state.confidence,
            metadata={'state_vector': awareness_state.vector.tolist()}
        )
```

### 2. Sensor Confidence Integration
```python
# awareness/core/sensor_bridge.py
from sensor_confidence.confidence_framework import ConfidenceFramework

class SensorConfidenceBridge:
    """Integrates with existing sensor confidence system"""
    
    def __init__(self):
        self.confidence_framework = ConfidenceFramework()
        
    def get_sensor_confidence(self, sensor_type: str) -> float:
        """Get current sensor confidence from framework"""
        return self.confidence_framework.get_confidence(sensor_type)
```

## Experimental Protocols

### 1. Ablation Studies
- Remove sensor fusion → measure awareness degradation
- Remove memory → test immediate reasoning only
- Remove CPTE routing → test self-contained knowledge
- Remove trust weights → test uniform sensor weighting

### 2. Scaling Studies
- Model size: 10M, 30M, 100M parameters
- Sensor count: 1, 4, 8, 16 sensors
- Memory window: 1s, 10s, 60s, 600s
- CPTE domains: 5, 20, 100, 1000

### 3. Robustness Studies
- Sensor failure: 0%, 25%, 50%, 75%
- Memory corruption: inject noise at various rates
- CPTE unavailability: simulate network failures
- Adversarial inputs: test decision stability

## Next Development Steps

1. **Week 1**: Implement core modules
   - [ ] Basic sensor fusion
   - [ ] Memory integration stub
   - [ ] CPTE router skeleton

2. **Week 2**: Data generation
   - [ ] Scenario generator
   - [ ] Sensor simulator
   - [ ] Knowledge gap identifier

3. **Week 3**: Training pipeline
   - [ ] Multi-objective loss functions
   - [ ] Distributed training setup
   - [ ] Checkpoint management

4. **Week 4**: Evaluation suite
   - [ ] Benchmark implementation
   - [ ] Ablation framework
   - [ ] Visualization tools

## Questions for Review

1. Should we prioritize certain sensors over others initially?
2. What's the preferred memory backend - SQLite or custom?
3. How many CPTE domains should we start with?
4. Any specific scenarios most relevant to your use cases?
5. Distributed training across multiple GPUs needed?

---

*Ready to build true situational awareness!*