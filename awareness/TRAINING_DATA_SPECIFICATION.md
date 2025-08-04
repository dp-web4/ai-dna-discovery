# HRM-CPTE Training Data Specification

## Overview

This document specifies the training data requirements for the HRM-CPTE integrated situational awareness system. The data must enable the model to learn:
1. Trust-weighted sensor fusion
2. Temporal memory patterns
3. Hierarchical reasoning
4. CPTE routing decisions

## Data Categories

### 1. Synthetic Awareness Scenarios

#### A. Navigation Tasks
```json
{
  "scenario_id": "nav_indoor_001",
  "environment": {
    "type": "indoor_office",
    "layout": "3x3_grid_rooms",
    "obstacles": ["furniture", "doors", "people"],
    "lighting": "variable",
    "noise_level": 0.3
  },
  "sensors": {
    "imu": {
      "data": [ax, ay, az, gx, gy, gz, mx, my, mz],
      "noise": 0.2,
      "drift": 0.05,
      "frequency": 100
    },
    "camera": {
      "resolution": [640, 480],
      "fps": 30,
      "occlusion": 0.15,
      "blur": 0.1
    },
    "audio": {
      "sample_rate": 16000,
      "snr": 20,
      "reverb": 0.3
    }
  },
  "task": {
    "goal": "navigate_to_room_9",
    "constraints": ["avoid_people", "minimize_time"],
    "knowledge_required": ["indoor_navigation", "social_norms"]
  },
  "temporal_sequence": [
    {"t": 0, "event": "start_in_room_1"},
    {"t": 5, "event": "door_blocked"},
    {"t": 10, "event": "person_approaching"},
    {"t": 15, "event": "lighting_change"}
  ],
  "ground_truth": {
    "path": [[1,1], [1,2], [2,2], [3,2], [3,3]],
    "decisions": ["turn_right", "wait", "proceed", "turn_left"],
    "cpte_calls": ["social_navigation_expert"]
  }
}
```

#### B. Object Identification
```json
{
  "scenario_id": "identify_medical_001",
  "environment": {
    "type": "medical_lab",
    "lighting": "fluorescent",
    "clutter_level": 0.7
  },
  "sensors": {
    "camera": {
      "views": ["top", "side"],
      "resolution": [1280, 720],
      "distortion": 0.05
    },
    "depth": {
      "range": [0.1, 5.0],
      "accuracy": 0.02
    }
  },
  "task": {
    "goal": "identify_medical_instruments",
    "constraints": ["high_accuracy", "explain_uncertainty"],
    "knowledge_required": ["medical_equipment", "safety_protocols"]
  },
  "objects": [
    {"id": "obj_1", "type": "scalpel", "position": [0.3, 0.4], "occluded": false},
    {"id": "obj_2", "type": "forceps", "position": [0.5, 0.6], "occluded": true},
    {"id": "obj_3", "type": "unknown_tool", "position": [0.7, 0.2], "occluded": false}
  ],
  "ground_truth": {
    "identifications": ["scalpel", "forceps", "unknown"],
    "confidence": [0.95, 0.60, 0.10],
    "cpte_calls": ["medical_instrument_expert"]
  }
}
```

### 2. Memory Integration Scenarios

#### A. Episodic Recall
```json
{
  "scenario_id": "memory_recall_001",
  "memory_sequence": [
    {"t": -300, "event": "entered_building", "location": "entrance"},
    {"t": -200, "event": "met_person", "person_id": "john_doe"},
    {"t": -100, "event": "received_instruction", "content": "go_to_room_305"},
    {"t": 0, "event": "current_query", "question": "where_did_i_meet_john"}
  ],
  "distractors": [
    {"t": -250, "event": "saw_poster", "irrelevant": true},
    {"t": -150, "event": "heard_announcement", "irrelevant": true}
  ],
  "ground_truth": {
    "answer": "entrance",
    "memory_type": "episodic",
    "confidence": 0.85
  }
}
```

#### B. Semantic Association
```json
{
  "scenario_id": "semantic_assoc_001",
  "context": "robotics_workshop",
  "query": "what_tools_needed_for_servo_repair",
  "memory_state": {
    "working": ["current_servo_model", "available_tools"],
    "episodic": ["previous_repair_experience"],
    "semantic": ["servo_repair_procedures", "tool_specifications"]
  },
  "ground_truth": {
    "answer": ["screwdriver_set", "multimeter", "soldering_iron"],
    "confidence": 0.90,
    "cpte_required": false
  }
}
```

### 3. CPTE Routing Scenarios

#### A. Knowledge Gap Detection
```json
{
  "scenario_id": "knowledge_gap_001",
  "situation": "chemical_spill_detected",
  "available_knowledge": {
    "internal": ["basic_safety", "evacuation_procedures"],
    "confidence": 0.3
  },
  "query": "proper_cleanup_procedure",
  "ground_truth": {
    "route_to_cpte": true,
    "cpte_domain": "hazmat_handling",
    "urgency": "high",
    "internal_confidence": 0.3
  }
}
```

#### B. Confidence Calibration
```json
{
  "scenario_id": "confidence_calib_001",
  "task": "translate_technical_document",
  "source_language": "japanese",
  "domain": "quantum_computing",
  "model_state": {
    "language_confidence": 0.6,
    "domain_confidence": 0.4,
    "combined_confidence": 0.24
  },
  "ground_truth": {
    "should_route": true,
    "cpte_type": "translation_expert",
    "expected_accuracy_internal": 0.45,
    "expected_accuracy_cpte": 0.92
  }
}
```

### 4. Multi-Modal Fusion Scenarios

#### A. Conflicting Sensors
```json
{
  "scenario_id": "sensor_conflict_001",
  "situation": "robot_localization",
  "sensor_readings": {
    "imu": {"position": [10.2, 5.3], "confidence": 0.7},
    "vision": {"position": [10.8, 5.1], "confidence": 0.8},
    "gps": {"position": [15.0, 8.0], "confidence": 0.3}
  },
  "environmental_factors": {
    "indoor": true,
    "metallic_interference": 0.6,
    "lighting": "good"
  },
  "ground_truth": {
    "true_position": [10.6, 5.2],
    "trust_weights": {"imu": 0.3, "vision": 0.6, "gps": 0.1},
    "fusion_strategy": "weighted_average"
  }
}
```

#### B. Temporal Sensor Degradation
```json
{
  "scenario_id": "sensor_degrade_001",
  "timeline": [
    {"t": 0, "camera_quality": 1.0, "imu_drift": 0.0},
    {"t": 60, "camera_quality": 0.8, "imu_drift": 0.1},
    {"t": 120, "camera_quality": 0.5, "imu_drift": 0.2},
    {"t": 180, "camera_quality": 0.2, "imu_drift": 0.3}
  ],
  "task": "maintain_accurate_position",
  "ground_truth": {
    "trust_evolution": [
      {"t": 0, "camera_trust": 0.8, "imu_trust": 0.2},
      {"t": 180, "camera_trust": 0.2, "imu_trust": 0.8}
    ],
    "adaptation_strategy": "dynamic_reweighting"
  }
}
```

## Data Generation Pipeline

### 1. Scenario Generator
```python
class ScenarioGenerator:
    def __init__(self):
        self.environments = EnvironmentLibrary()
        self.sensor_models = SensorSimulator()
        self.task_generator = TaskGenerator()
        
    def generate_batch(self, size=1000, difficulty=0.5):
        scenarios = []
        for _ in range(size):
            env = self.environments.sample(difficulty)
            sensors = self.sensor_models.configure(env)
            task = self.task_generator.create(env, difficulty)
            
            scenario = self.simulate_scenario(env, sensors, task)
            scenarios.append(scenario)
            
        return scenarios
```

### 2. Noise and Corruption Models
```python
class NoiseModels:
    @staticmethod
    def sensor_noise(data, noise_level):
        # Realistic sensor noise patterns
        if sensor_type == "imu":
            return add_gaussian_noise(data, sigma=noise_level)
        elif sensor_type == "camera":
            return add_blur_and_occlusion(data, level=noise_level)
            
    @staticmethod
    def memory_corruption(memory, corruption_rate):
        # Simulate memory decay and interference
        corrupted = memory.copy()
        mask = np.random.random(len(memory)) < corruption_rate
        corrupted[mask] = add_interference(memory[mask])
        return corrupted
```

### 3. Ground Truth Generation
```python
class GroundTruthGenerator:
    def __init__(self):
        self.optimal_solver = OptimalPolicyNetwork()
        self.human_validator = HumanValidationInterface()
        
    def generate_ground_truth(self, scenario):
        # Automated optimal solution
        auto_solution = self.optimal_solver.solve(scenario)
        
        # Human validation for complex cases
        if scenario.complexity > 0.8:
            human_solution = self.human_validator.validate(
                scenario, auto_solution
            )
            return human_solution
            
        return auto_solution
```

## Data Validation Requirements

### 1. Scenario Completeness
- All sensor modalities present
- Temporal sequences properly ordered
- Ground truth covers all decision points
- CPTE routing decisions justified

### 2. Realism Checks
- Sensor noise within realistic bounds
- Environmental conditions physically plausible
- Task requirements achievable
- Temporal dynamics consistent

### 3. Balance Metrics
- 40% navigation tasks
- 30% identification tasks  
- 20% knowledge routing
- 10% edge cases

### 4. Difficulty Distribution
- 20% easy (tutorial level)
- 50% medium (standard operation)
- 20% hard (challenging conditions)
- 10% extreme (stress testing)

## Dataset Statistics

### Target Dataset Size
- **Training**: 100,000 scenarios
- **Validation**: 20,000 scenarios
- **Test**: 20,000 scenarios
- **Edge Cases**: 5,000 scenarios

### Computational Requirements
- **Generation Time**: ~48 hours on 8 GPUs
- **Storage**: ~500GB uncompressed
- **Memory**: 32GB RAM for generation

### Quality Metrics
- **Human Agreement**: >85% on ground truth
- **Sensor Realism**: Validated against real hardware
- **Task Diversity**: >1000 unique task types
- **Environmental Coverage**: >100 environment types

## Integration with Existing Data

### 1. AI DNA Discovery Datasets
- Consciousness notation examples
- Multi-model collaboration logs
- Distributed memory patterns

### 2. Sensor Confidence Data
- Real sensor measurements from IMU/camera
- Confidence calibration curves
- Failure mode examples

### 3. Memory System Logs
- SQLite query patterns
- Recall accuracy statistics
- Compression ratios

## Next Steps

1. Implement scenario generators
2. Validate against real sensor data
3. Generate initial 10K examples
4. Human validation of ground truth
5. Scale to full dataset

---

*"Good data is the foundation of good reasoning. Great data includes the edge cases where reasoning breaks down."*