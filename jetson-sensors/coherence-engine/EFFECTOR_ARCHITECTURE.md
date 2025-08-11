# Effector Architecture for Coherence Engine

*August 11, 2025*
*Jetson Implementation Specification*

## Overview

Effectors are the action-generating counterpart to sensors in the Coherence Engine. Just as reality emerges from weighted sensor fusion, **action emerges from weighted effector fusion**. This document specifies the effector implementation for the Jetson platform.

## Core Architecture

### Base Effector Class

```python
class EffectorBase:
    """Base class for all effectors - mirrors SensorBase"""
    
    def __init__(self, effector_id, mrh_context):
        self.id = effector_id
        self.mrh = mrh_context  # Markov Relevancy Horizon
        self.influence_weight = 1.0  # How much this effector can change reality
        self.confidence = 1.0  # How certain we are of outcomes
        self.energy_cost = 0.0  # Computational/physical energy required
        self.latency_ms = 0  # Time until effect manifests
        self.last_action = None
        self.last_outcome = None
        
    def propose_action(self, reality_field, goal_state):
        """Given current reality and goal, propose an action"""
        raise NotImplementedError
        
    def execute(self, action):
        """Actually perform the action"""
        raise NotImplementedError
        
    def predict_outcome(self, action):
        """Predict reality field after action"""
        raise NotImplementedError
        
    def update_confidence(self, predicted, actual):
        """Update confidence based on prediction accuracy"""
        error = self.compute_error(predicted, actual)
        self.confidence *= (1.0 - error * 0.1)  # Decay on error
        self.confidence = max(0.1, min(1.0, self.confidence))
```

## Jetson-Specific Effectors

### 1. Display Effector

```python
class DisplayEffector(EffectorBase):
    """Visual output to HDMI/screen"""
    
    def __init__(self):
        super().__init__("display", "visual_output")
        self.energy_cost = 0.02  # Low computational cost
        self.latency_ms = 16  # One frame at 60Hz
        
    def execute(self, action):
        # action = {'type': 'overlay', 'data': coherence_state}
        # or action = {'type': 'attention_box', 'coords': [x,y,w,h]}
        cv2.putText(frame, action['data'], ...)
```

### 2. GPIO Effector

```python
class GPIOEffector(EffectorBase):
    """Control GPIO pins for LEDs, motors, etc."""
    
    def __init__(self, pin_config):
        super().__init__("gpio", "physical_output")
        self.pins = pin_config
        self.energy_cost = 0.01  # Minimal
        self.latency_ms = 1
        
    def execute(self, action):
        # action = {'pin': 7, 'state': 'HIGH'}
        GPIO.output(action['pin'], action['state'])
```

### 3. Speech Effector

```python
class SpeechEffector(EffectorBase):
    """Text-to-speech output"""
    
    def __init__(self):
        super().__init__("speech", "audio_output")
        self.energy_cost = 0.05  # TTS processing
        self.latency_ms = 200  # TTS generation time
        
    def execute(self, action):
        # action = {'text': 'Hello world', 'emotion': 'neutral'}
        subprocess.run(['espeak', action['text']])
```

### 4. Memory Write Effector

```python
class MemoryWriteEffector(EffectorBase):
    """Commits experiences to memory"""
    
    def __init__(self, memory_sensor):
        super().__init__("memory_write", "temporal_output")
        self.memory = memory_sensor
        self.energy_cost = 0.03
        self.latency_ms = 10
        
    def execute(self, action):
        # action = {'experience': {...}, 'importance': 0.8}
        self.memory.commit(action['experience'], action['importance'])
```

### 5. Network Effector

```python
class NetworkEffector(EffectorBase):
    """Send messages to other nodes (Legion, etc.)"""
    
    def __init__(self, bridge_config):
        super().__init__("network", "distributed_output")
        self.bridge = bridge_config
        self.energy_cost = 0.04
        self.latency_ms = 50  # Network latency
        
    def execute(self, action):
        # action = {'target': 'legion', 'message': {...}}
        self.bridge.send(action['target'], action['message'])
```

## Action Field Generation

```python
class ActionField:
    """Generates and fuses possible actions"""
    
    def __init__(self, effectors, context):
        self.effectors = effectors
        self.context = context
        
    def generate(self, reality_field, goal_state):
        """Generate weighted action proposals"""
        proposals = {}
        
        for effector in self.effectors:
            if not self.is_relevant(effector, goal_state):
                continue
                
            action = effector.propose_action(reality_field, goal_state)
            proposals[effector.id] = {
                'action': action,
                'influence': effector.influence_weight,
                'confidence': effector.confidence,
                'cost': effector.energy_cost,
                'latency': effector.latency_ms,
                'predicted_outcome': effector.predict_outcome(action)
            }
        
        return self.fuse_actions(proposals)
    
    def fuse_actions(self, proposals):
        """Combine proposals into coherent action plan"""
        # Weight by (influence * confidence) / (cost + latency/1000)
        weighted_actions = []
        
        for eff_id, proposal in proposals.items():
            weight = (proposal['influence'] * proposal['confidence']) / \
                    (proposal['cost'] + proposal['latency']/1000 + 0.01)
            weighted_actions.append({
                'effector_id': eff_id,
                'action': proposal['action'],
                'weight': weight,
                'predicted_outcome': proposal['predicted_outcome']
            })
        
        return sorted(weighted_actions, key=lambda x: x['weight'], reverse=True)
```

## Action Selection

```python
class CoherenceActionSelector:
    """Selects actions that maintain/increase coherence"""
    
    def __init__(self, coherence_engine):
        self.engine = coherence_engine
        self.energy_budget = 1.0  # Per cycle budget
        
    def select_action(self, reality_field, action_field, goal_state):
        """Select best action within energy budget"""
        available_energy = self.energy_budget
        selected_actions = []
        
        for action in action_field:
            # Predict coherence after action
            predicted_reality = self.simulate_outcome(
                reality_field, action
            )
            
            # Measure coherence with goal
            coherence_delta = self.compute_coherence_delta(
                reality_field, predicted_reality, goal_state
            )
            
            # Check energy budget
            if action['cost'] <= available_energy:
                if coherence_delta > 0:  # Improves coherence
                    selected_actions.append(action)
                    available_energy -= action['cost']
        
        return selected_actions
```

## Integration with Coherence Engine

```python
class JetsonCoherenceEngine:
    """Extended with effector support"""
    
    def __init__(self):
        # ... existing sensor setup ...
        
        # Initialize effectors
        self.effectors = {
            'display': DisplayEffector(),
            'gpio': GPIOEffector(pin_config),
            'speech': SpeechEffector(),
            'memory': MemoryWriteEffector(self.memory_sensor),
            'network': NetworkEffector(bridge_config)
        }
        
        self.action_field = ActionField(self.effectors, self.context)
        self.action_selector = CoherenceActionSelector(self)
        
    def cycle(self):
        """Main perception-action loop"""
        # Sense
        reality = self.generate_reality_field()
        
        # Determine goal
        goal = self.determine_goal(reality)
        
        # Generate actions
        actions = self.action_field.generate(reality, goal)
        
        # Select actions
        selected = self.action_selector.select_action(
            reality, actions, goal
        )
        
        # Execute
        for action in selected:
            effector = self.effectors[action['effector_id']]
            effector.execute(action['action'])
        
        # Learn from outcome
        new_reality = self.generate_reality_field()
        self.update_weights(reality, selected, new_reality)
```

## Energy Management

```python
class EnergyManager:
    """Tracks and allocates energy across effectors"""
    
    def __init__(self, total_budget=1.0):
        self.total_budget = total_budget
        self.used = 0.0
        self.history = []
        
    def allocate(self, effector, action):
        """Check if action is within budget"""
        cost = effector.energy_cost * action.intensity
        if self.used + cost <= self.total_budget:
            self.used += cost
            self.history.append({
                'time': time.time(),
                'effector': effector.id,
                'cost': cost
            })
            return True
        return False
    
    def reset_cycle(self):
        """Reset for next cycle"""
        self.used = 0.0
```

## Testing

### Basic Effector Test

```python
def test_display_effector():
    """Test visual output"""
    eff = DisplayEffector()
    
    # Test attention box
    action = {
        'type': 'attention_box',
        'coords': [100, 100, 50, 50],
        'color': (0, 255, 0)
    }
    eff.execute(action)
    
    # Test overlay
    action = {
        'type': 'overlay',
        'data': 'Coherence: 0.87',
        'position': (10, 30)
    }
    eff.execute(action)
```

### Integration Test

```python
def test_perception_action_loop():
    """Test full cycle"""
    engine = JetsonCoherenceEngine()
    
    for i in range(100):
        engine.cycle()
        time.sleep(0.033)  # 30 Hz
        
        # Log energy usage
        print(f"Cycle {i}: Energy used: {engine.energy_manager.used:.3f}")
```

## Configuration

```yaml
# effector_config.yaml
effectors:
  display:
    enabled: true
    energy_cost: 0.02
    max_overlays: 5
    
  gpio:
    enabled: true
    pins:
      - {id: 7, name: "status_led", mode: "output"}
      - {id: 11, name: "attention_led", mode: "output"}
    
  speech:
    enabled: false  # Enable when audio ready
    engine: "espeak"
    rate: 150
    
  memory:
    enabled: true
    max_commits_per_cycle: 3
    
  network:
    enabled: true
    targets: ["legion", "cloud"]
    max_messages_per_cycle: 5
```

## Next Steps

1. Implement basic effectors (display, GPIO)
2. Test perception-action loop at 30 Hz
3. Add energy tracking and visualization
4. Implement goal generation (homeostatic + directive)
5. Add learning from action outcomes
6. Connect to Legion via network effector

---

*This architecture enables the Jetson to not just perceive reality but actively shape it through coherent action.*