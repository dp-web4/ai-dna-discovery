#!/usr/bin/env python3
"""
Sleep Cycle Implementation for Coherence Engine
Sleep as mandatory maintenance for the memory temporal sensor
"""

import time
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque

@dataclass
class SleepMetrics:
    """Metrics that determine when sleep is needed"""
    retrieval_latency: float  # Average time to retrieve memories
    pattern_accuracy: float   # Success rate of pattern matching
    memory_pressure: float    # Used capacity / total capacity
    trust_staleness: float    # Hours since last calibration
    experience_backlog: int   # Number of unprocessed experiences
    time_awake: float        # Hours since last sleep
    
    def sleep_urgency(self) -> float:
        """Calculate urgency score 0.0 (fresh) to 1.0 (must sleep)"""
        scores = [
            min(self.retrieval_latency / 1.0, 1.0),  # >1s is bad
            1.0 - self.pattern_accuracy,  # Lower accuracy = higher urgency
            self.memory_pressure,  # Direct correlation
            min(self.trust_staleness / 24.0, 1.0),  # >24h is critical
            min(self.experience_backlog / 1000, 1.0),  # >1000 is critical
            min(self.time_awake / 16.0, 1.0)  # >16h needs sleep
        ]
        # Weighted average with emphasis on time awake and memory pressure
        weights = [0.15, 0.15, 0.2, 0.1, 0.15, 0.25]
        return sum(s * w for s, w in zip(scores, weights))

@dataclass
class DreamScenario:
    """A test scenario generated during REM sleep"""
    timestamp: str
    elements: List[Dict]
    mutations: List[str]
    emotional_amplitude: float
    physical_constraints_relaxed: bool
    validation_result: Optional[Dict] = None

class SleepCycle:
    """Manages sleep cycles for the Coherence Engine"""
    
    def __init__(self, memory_sensor, coherence_engine, sleep_dir="memory/sleep"):
        self.memory = memory_sensor
        self.engine = coherence_engine
        self.sleep_dir = Path(sleep_dir)
        self.sleep_dir.mkdir(parents=True, exist_ok=True)
        
        # Sleep configuration
        self.wake_duration = timedelta(hours=16)
        self.sleep_duration = timedelta(hours=8)
        self.last_sleep = datetime.now()
        self.is_sleeping = False
        
        # Sleep stage durations (proportional to 8-hour sleep)
        self.stage_durations = {
            'light_1': timedelta(hours=1),
            'deep': timedelta(hours=3),
            'rem': timedelta(hours=3),
            'light_2': timedelta(hours=1)
        }
        
        # Dream log
        self.dream_log = deque(maxlen=100)
        
    def get_metrics(self) -> SleepMetrics:
        """Calculate current sleep metrics"""
        time_awake = (datetime.now() - self.last_sleep).total_seconds() / 3600
        
        # Get metrics from memory sensor
        # Estimate retrieval latency based on memory size
        retrieval_latency = min(1.0, len(self.memory.experiences) / 5000.0) if hasattr(self.memory, 'experiences') else 0.1
        # Use field stability as proxy for pattern accuracy
        pattern_accuracy = self.memory.patterns.get('field_stability', 0.5) if hasattr(self.memory, 'patterns') else 0.5
        # Calculate memory pressure
        memory_pressure = len(self.memory.experiences) / self.memory.max_memory if hasattr(self.memory, 'max_memory') else 0.5
        # Estimate trust staleness
        trust_staleness = min(24.0, len(self.memory.experiences) / 100.0) if hasattr(self.memory, 'experiences') else 1.0
        # Count working memory as backlog
        experience_backlog = len(self.memory.working_memory) if hasattr(self.memory, 'working_memory') else 0
        
        return SleepMetrics(
            retrieval_latency=retrieval_latency,
            pattern_accuracy=pattern_accuracy,
            memory_pressure=memory_pressure,
            trust_staleness=trust_staleness,
            experience_backlog=experience_backlog,
            time_awake=time_awake
        )
    
    def should_sleep(self) -> bool:
        """Determine if sleep is needed"""
        metrics = self.get_metrics()
        urgency = metrics.sleep_urgency()
        
        # Log metrics
        self.log_metrics(metrics, urgency)
        
        # Sleep if urgency > 0.7 or been awake too long
        return urgency > 0.7 or metrics.time_awake > 20
    
    def enter_sleep(self):
        """Begin sleep cycle"""
        print("\n🌙 Entering sleep cycle...")
        self.is_sleeping = True
        sleep_start = datetime.now()
        
        # Save pre-sleep state
        self.save_state("pre_sleep")
        
        # Pause external sensors
        original_context = self.engine.context
        self.engine.pause_external_sensors()
        
        # Execute sleep stages
        stages = [
            ("Light Sleep (N1/N2)", self.light_sleep_1),
            ("Deep Sleep (N3)", self.deep_sleep),
            ("REM Sleep", self.rem_sleep),
            ("Light Sleep (N2)", self.light_sleep_2)
        ]
        
        for stage_name, stage_func in stages:
            print(f"\n💤 {stage_name}...")
            stage_func()
            time.sleep(0.5)  # Brief pause between stages
        
        # Wake up
        self.wake_up(original_context)
        
        # Record sleep duration
        sleep_duration = datetime.now() - sleep_start
        print(f"\n☀️ Woke up after {sleep_duration.total_seconds():.1f} seconds of processing")
        
        self.is_sleeping = False
        self.last_sleep = datetime.now()
    
    def light_sleep_1(self):
        """Initial sorting and organization"""
        print("  📂 Sorting recent experiences...")
        
        # Sort experiences by relevance and recency
        if hasattr(self.memory, 'working_memory'):
            experiences = list(self.memory.working_memory)[-100:]  # Get recent experiences
            sorted_exp = sorted(experiences, 
                              key=lambda x: x.field_value if hasattr(x, 'field_value') else 0.5,
                              reverse=True)
            # In real implementation, would stage for processing
            processed_count = min(100, len(sorted_exp))
        else:
            processed_count = 0
        
        print(f"  ✓ Staged {processed_count} experiences for processing")
    
    def deep_sleep(self):
        """Heavy processing: pattern extraction, index rebuilding, pruning"""
        print("  🔬 Extracting patterns...")
        
        # Pattern extraction
        patterns = self.extract_patterns() if hasattr(self, 'extract_patterns') else []
        if patterns:
            self.store_patterns(patterns) if hasattr(self, 'store_patterns') else None
            print(f"  ✓ Extracted {len(patterns)} patterns")
        
        # Rebuild retrieval index
        print("  🗂️ Rebuilding memory index...")
        if hasattr(self.memory, '_update_patterns'):
            self.memory._update_patterns()
        elif hasattr(self, 'rebuild_index'):
            self.rebuild_index()
        
        # Recalibrate trust weights
        print("  ⚖️ Recalibrating trust weights...")
        self.recalibrate_trust()
        
        # Prune irrelevant memories
        print("  🧹 Pruning irrelevant memories...")
        pruned = self.prune_irrelevant(threshold=0.1) if hasattr(self, 'prune_irrelevant') else 0
        print(f"  ✓ Pruned {pruned} low-relevance memories")
    
    def rem_sleep(self):
        """Test patterns through dream simulation"""
        print("  💭 Entering REM sleep (dream simulation)...")
        
        # Generate and test dream scenarios
        num_dreams = random.randint(3, 7)
        for i in range(num_dreams):
            scenario = self.generate_dream_scenario()
            validation = self.test_dream_scenario(scenario)
            scenario.validation_result = validation
            
            # Log dream
            self.dream_log.append(scenario)
            self.save_dream(scenario)
            
            print(f"    💫 Dream {i+1}: {scenario.mutations[0] if scenario.mutations else 'normal'}")
    
    def light_sleep_2(self):
        """Final integration and preparation for waking"""
        print("  🔄 Integrating processed memories...")
        
        # Consolidate similar experiences
        consolidated = self.consolidate_similar() if hasattr(self, 'consolidate_similar') else 0
        print(f"  ✓ Consolidated {consolidated} similar memories")
        
        # Prepare memory for waking state
        if hasattr(self, 'prepare_for_wake'):
            self.prepare_for_wake()
        elif hasattr(self.memory, '_update_patterns'):
            self.memory._update_patterns()
        
        # Save post-sleep state
        self.save_state("post_sleep")
    
    def wake_up(self, original_context):
        """Gradually rebuild reality field"""
        print("\n🌅 Waking up...")
        
        # Grogginess phase - gradual reactivation
        print("  😴 Reactivating sensors gradually...")
        if hasattr(self.engine, 'resume_external_sensors'):
            self.engine.resume_external_sensors(gradual=True)
        elif hasattr(self, 'resume_external_sensors'):
            self.resume_external_sensors(gradual=True)
        
        # Rebuild reality field with processed memory
        print("  🧠 Rebuilding reality field...")
        if hasattr(self, 'reinitialize_memory'):
            self.reinitialize_memory()
        elif hasattr(self.memory, '_update_patterns'):
            self.memory._update_patterns()
        
        # Restore context
        self.engine.context = original_context
        print("  ✓ Reality field restored")
    
    def generate_dream_scenario(self) -> DreamScenario:
        """Generate a test scenario for pattern validation"""
        # Get recent experiences and patterns
        recent = list(self.memory.working_memory)[-10:] if hasattr(self.memory, 'working_memory') else []
        patterns = list(self.memory.patterns.items())[:5] if hasattr(self.memory, 'patterns') else []
        
        # Combine with mutations
        elements = recent[:5] + patterns[:3]
        
        # Apply dream logic (unusual combinations)
        mutations = []
        if random.random() > 0.5:
            mutations.append("time_dilation")
        if random.random() > 0.5:
            mutations.append("identity_fluidity")
        if random.random() > 0.5:
            mutations.append("physics_violation")
        if random.random() > 0.7:
            mutations.append("emotional_amplification")
        
        return DreamScenario(
            timestamp=datetime.now().isoformat(),
            elements=elements,
            mutations=mutations,
            emotional_amplitude=random.uniform(0.5, 2.0),
            physical_constraints_relaxed=len(mutations) > 0
        )
    
    def test_dream_scenario(self, scenario: DreamScenario) -> Dict:
        """Test a scenario through the coherence engine"""
        # Run simulation with dream scenario
        result = {'coherence': 0.5}  # Simplified simulation
        if hasattr(self, 'engine') and hasattr(self.engine, 'simulate_scenario'):
            result = self.engine.simulate_scenario(scenario.elements)
        
        # Validate patterns
        validation = {
            'coherence': result.get('coherence', 0),
            'patterns_confirmed': [],
            'patterns_violated': [],
            'new_patterns_discovered': []
        }
        
        # Check which patterns held up in the dream
        if hasattr(self.memory, 'patterns'):
            for pattern_key in self.memory.patterns:
                pattern = {'id': pattern_key}
                if self.pattern_matches_scenario(pattern, scenario):
                    validation['patterns_confirmed'].append(pattern['id'])
                else:
                    validation['patterns_violated'].append(pattern['id'])
        
        return validation
    
    def pattern_matches_scenario(self, pattern: Dict, scenario: DreamScenario) -> bool:
        """Check if a pattern is validated by the dream scenario"""
        # Simplified pattern matching
        # In reality, this would be more sophisticated
        return random.random() > 0.3  # 70% of patterns are confirmed
    
    def extract_patterns(self) -> List[Dict]:
        """Extract patterns from memory experiences"""
        patterns = []
        if not hasattr(self.memory, 'experiences') or len(self.memory.experiences) < 10:
            return patterns
        
        experiences = list(self.memory.experiences)[-100:]  # Last 100
        
        # Pattern: State transitions
        transitions = {}
        for i in range(len(experiences) - 1):
            if hasattr(experiences[i], 'context_state'):
                curr = experiences[i].context_state
                next = experiences[i + 1].context_state
                key = f"{curr}->{next}"
                transitions[key] = transitions.get(key, 0) + 1
        
        if transitions:
            patterns.append({
                'id': 'transitions',
                'type': 'state_transitions',
                'data': transitions
            })
        
        return patterns
    
    def store_patterns(self, patterns: List[Dict]):
        """Store extracted patterns"""
        pattern_file = self.sleep_dir / f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(pattern_file, 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
    
    def rebuild_index(self):
        """Rebuild memory retrieval index"""
        if hasattr(self.memory, '_update_patterns'):
            self.memory._update_patterns()
    
    def prune_irrelevant(self, threshold: float = 0.1) -> int:
        """Remove low-relevance memories"""
        if not hasattr(self.memory, 'experiences'):
            return 0
        
        initial_count = len(self.memory.experiences)
        kept = []
        
        for exp in self.memory.experiences:
            # Keep if has trigger or significant field value
            if hasattr(exp, 'trigger') and exp.trigger:
                kept.append(exp)
            elif hasattr(exp, 'field_value') and abs(exp.field_value - 0.5) > threshold:
                kept.append(exp)
        
        if hasattr(self.memory, 'max_memory'):
            kept = kept[-self.memory.max_memory:]
        
        self.memory.experiences.clear()
        self.memory.experiences.extend(kept)
        
        return initial_count - len(kept)
    
    def consolidate_similar(self) -> int:
        """Consolidate similar consecutive experiences"""
        if not hasattr(self.memory, 'experiences') or len(self.memory.experiences) < 3:
            return 0
        
        consolidated = 0
        new_exp = []
        experiences = list(self.memory.experiences)
        
        i = 0
        while i < len(experiences):
            current = experiences[i]
            similar_group = [current]
            
            # Find similar consecutive experiences
            j = i + 1
            while j < len(experiences) and j < i + 5:
                next_exp = experiences[j]
                if (hasattr(current, 'context_state') and hasattr(next_exp, 'context_state') and
                    current.context_state == next_exp.context_state and
                    hasattr(current, 'field_value') and hasattr(next_exp, 'field_value') and
                    abs(current.field_value - next_exp.field_value) < 0.1):
                    similar_group.append(next_exp)
                    j += 1
                else:
                    break
            
            if len(similar_group) > 2:
                # Keep first, note consolidation
                if hasattr(current, 'notes'):
                    current.notes['consolidated'] = len(similar_group)
                new_exp.append(current)
                consolidated += len(similar_group) - 1
                i = j
            else:
                new_exp.append(current)
                i += 1
        
        self.memory.experiences.clear()
        self.memory.experiences.extend(new_exp)
        
        return consolidated
    
    def prepare_for_wake(self):
        """Prepare memory for waking"""
        if hasattr(self.memory, 'working_memory'):
            self.memory.working_memory.clear()
            recent = list(self.memory.experiences)[-50:] if hasattr(self.memory, 'experiences') else []
            for exp in recent:
                self.memory.working_memory.append(exp)
        
        if hasattr(self.memory, '_update_patterns'):
            self.memory._update_patterns()
    
    def resume_external_sensors(self, gradual: bool = True):
        """Resume external sensor operations"""
        # Placeholder for sensor resumption
        pass
    
    def reinitialize_memory(self):
        """Reinitialize memory after sleep"""
        if hasattr(self.memory, '_update_patterns'):
            self.memory._update_patterns()
    
    def calculate_prediction_accuracies(self) -> Dict[str, float]:
        """Calculate prediction accuracy for trust recalibration"""
        # Simple implementation - would be more sophisticated in production
        return {'memory': 0.8, 'vision': 0.7, 'imu': 0.75}
    
    def recalibrate_trust(self):
        """Recalibrate trust weights based on prediction accuracy"""
        # Get prediction accuracy from recent experiences
        accuracies = self.calculate_prediction_accuracies() if hasattr(self, 'calculate_prediction_accuracies') else {}
        
        # Update trust weights
        for sensor_id, accuracy in accuracies.items():
            old_trust = self.memory.trust_weights.get(sensor_id, 0.5)
            # Gradual trust adjustment
            new_trust = old_trust * 0.8 + accuracy * 0.2
            self.memory.trust_weights[sensor_id] = new_trust
    
    def save_state(self, state_type: str):
        """Save sleep cycle state"""
        state_file = self.sleep_dir / f"{state_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        state = {
            'type': state_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(self.get_metrics()),
            'memory_size': len(self.memory.experiences) if hasattr(self.memory, 'experiences') else 0,
            'pattern_count': len(self.memory.patterns) if hasattr(self.memory, 'patterns') else 0,
            'trust_weights': self.memory.trust_weights if hasattr(self.memory, 'trust_weights') else {}
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def save_dream(self, dream: DreamScenario):
        """Save dream scenario to disk"""
        dream_file = self.sleep_dir / f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(dream_file, 'w') as f:
            json.dump(asdict(dream), f, indent=2, default=str)
    
    def log_metrics(self, metrics: SleepMetrics, urgency: float):
        """Log sleep metrics"""
        log_file = self.sleep_dir / "sleep_metrics.jsonl"
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(metrics),
            'urgency': urgency,
            'should_sleep': urgency > 0.7
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

class SleepMonitor:
    """Monitor that triggers sleep cycles when needed"""
    
    def __init__(self, sleep_cycle: SleepCycle, check_interval: int = 60):
        self.sleep_cycle = sleep_cycle
        self.check_interval = check_interval  # seconds
        self.running = False
    
    def start(self):
        """Start monitoring for sleep needs"""
        self.running = True
        print("🛏️ Sleep monitor started")
        
        while self.running:
            if not self.sleep_cycle.is_sleeping:
                if self.sleep_cycle.should_sleep():
                    print("\n⚠️ Sleep needed - entering sleep cycle")
                    self.sleep_cycle.enter_sleep()
            
            time.sleep(self.check_interval)
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        print("🛏️ Sleep monitor stopped")

if __name__ == "__main__":
    # Example usage
    print("Sleep Cycle Implementation")
    print("This would be integrated with the Coherence Engine")
    print("\nKey concepts:")
    print("- Sleep as memory sensor maintenance")
    print("- Pattern extraction during deep sleep")
    print("- Dream scenarios for pattern validation")
    print("- Trust weight recalibration")
    print("- Gradual wake-up with reality field rebuilding")