#!/usr/bin/env python3
"""
Integration of Sleep Cycle with Coherence Engine
Connects the sleep cycle system to the actual running coherence engine
"""

import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import threading
import signal
import sys

# Import the coherence engine components
from coherence_engine import CoherenceEngine, ContextState
from sensors.persistent_memory_sensor import PersistentMemorySensor
from sensors.real_vision_sensor import RealVisionSensor
from sensors.real_imu_sensor import RealIMUSensor
from sleep_cycle import SleepCycle, SleepMonitor, SleepMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger("sleep_integration")

class IntegratedSleepCycle(SleepCycle):
    """Extended sleep cycle that integrates with real coherence engine"""
    
    def __init__(self, coherence_engine, memory_sensor, sleep_dir="memory/sleep"):
        # Store the actual engine and memory sensor
        self.engine = coherence_engine
        self.memory = memory_sensor
        
        # Initialize base sleep cycle
        self.sleep_dir = Path(sleep_dir)
        self.sleep_dir.mkdir(parents=True, exist_ok=True)
        
        # Sleep configuration
        self.wake_duration = timedelta(hours=16)
        self.sleep_duration = timedelta(hours=8)
        self.last_sleep = datetime.now()
        self.is_sleeping = False
        
        # Sleep stage durations (scaled for testing - minutes instead of hours)
        self.stage_durations = {
            'light_1': timedelta(minutes=2),
            'deep': timedelta(minutes=5),
            'rem': timedelta(minutes=5),
            'light_2': timedelta(minutes=2)
        }
        
        # Pattern storage
        self.extracted_patterns = []
        self.dream_scenarios = []
        
    def get_metrics(self) -> SleepMetrics:
        """Calculate current sleep metrics from actual sensors"""
        time_awake = (datetime.now() - self.last_sleep).total_seconds() / 3600
        
        # Calculate metrics from actual memory sensor
        retrieval_latency = self._calculate_retrieval_latency()
        pattern_accuracy = self._calculate_pattern_accuracy()
        memory_pressure = len(self.memory.experiences) / self.memory.max_memory
        trust_staleness = self._calculate_trust_staleness()
        experience_backlog = len(self.memory.working_memory)
        
        return SleepMetrics(
            retrieval_latency=retrieval_latency,
            pattern_accuracy=pattern_accuracy,
            memory_pressure=memory_pressure,
            trust_staleness=trust_staleness,
            experience_backlog=experience_backlog,
            time_awake=time_awake
        )
    
    def _calculate_retrieval_latency(self) -> float:
        """Measure how long it takes to retrieve similar experiences"""
        if len(self.memory.experiences) < 10:
            return 0.1  # Fast when few memories
        
        # Simulate retrieval time based on memory size
        # In real implementation, would measure actual retrieval
        return min(1.0, len(self.memory.experiences) / 5000.0)
    
    def _calculate_pattern_accuracy(self) -> float:
        """Calculate how well patterns predict outcomes"""
        if not self.memory.patterns:
            return 0.5  # Neutral when no patterns
        
        # Use field stability as proxy for pattern accuracy
        return self.memory.patterns.get("field_stability", 0.5)
    
    def _calculate_trust_staleness(self) -> float:
        """Hours since trust weights were updated"""
        # For testing, use minutes instead of hours
        # In real system, track actual update times
        return min(24.0, len(self.memory.experiences) / 100.0)
    
    def pause_external_sensors(self):
        """Pause real-time sensor updates during sleep"""
        logger.info("Pausing external sensors for sleep cycle")
        # Store current sensor states
        self.stored_sensor_states = {}
        for sensor_id in self.engine.sensors:
            # Could implement actual pause if sensors support it
            self.stored_sensor_states[sensor_id] = True
    
    def resume_external_sensors(self, gradual=True):
        """Resume sensor operations after sleep"""
        logger.info(f"Resuming external sensors (gradual={gradual})")
        if gradual:
            # Gradually increase sensor trust over a few ticks
            for sensor_id in self.engine.sensors:
                current_trust = self.engine.trust_model.base.get(sensor_id, 0.5)
                # Start at 50% trust and gradually increase
                self.engine.trust_model.base[sensor_id] = current_trust * 0.5
    
    def extract_patterns(self) -> list:
        """Extract patterns from accumulated experiences"""
        patterns = []
        
        if len(self.memory.experiences) < 10:
            return patterns
        
        # Extract context transition patterns
        experiences = list(self.memory.experiences)[-100:]  # Last 100 experiences
        
        # Pattern 1: State transition frequencies
        transitions = {}
        for i in range(len(experiences) - 1):
            curr_state = experiences[i].context_state
            next_state = experiences[i + 1].context_state
            key = f"{curr_state}->{next_state}"
            transitions[key] = transitions.get(key, 0) + 1
        
        if transitions:
            patterns.append({
                'type': 'state_transitions',
                'data': transitions,
                'confidence': 0.8
            })
        
        # Pattern 2: Trigger patterns
        triggers = [exp.trigger for exp in experiences if exp.trigger]
        if triggers:
            trigger_freq = {}
            for trigger in triggers:
                trigger_freq[trigger] = trigger_freq.get(trigger, 0) + 1
            patterns.append({
                'type': 'trigger_frequencies',
                'data': trigger_freq,
                'confidence': 0.7
            })
        
        # Pattern 3: Field value ranges per context
        context_fields = {}
        for exp in experiences:
            if exp.context_state not in context_fields:
                context_fields[exp.context_state] = []
            context_fields[exp.context_state].append(exp.field_value)
        
        field_patterns = {}
        for context, fields in context_fields.items():
            if fields:
                import numpy as np
                field_patterns[context] = {
                    'mean': np.mean(fields),
                    'std': np.std(fields),
                    'min': min(fields),
                    'max': max(fields)
                }
        
        if field_patterns:
            patterns.append({
                'type': 'field_ranges',
                'data': field_patterns,
                'confidence': 0.9
            })
        
        self.extracted_patterns = patterns
        return patterns
    
    def store_patterns(self, patterns):
        """Store extracted patterns for future use"""
        import json
        pattern_file = self.sleep_dir / f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(pattern_file, 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
        logger.info(f"Stored {len(patterns)} patterns to {pattern_file}")
    
    def rebuild_index(self):
        """Optimize memory retrieval indices"""
        logger.info("Rebuilding memory retrieval index...")
        # In a real implementation, this would rebuild data structures
        # for efficient similarity search, possibly using:
        # - KD-trees for spatial indexing
        # - LSH for high-dimensional similarity
        # - Inverted indices for context lookups
        self.memory._update_patterns()
    
    def prune_irrelevant(self, threshold=0.1):
        """Remove low-relevance memories to save space"""
        initial_count = len(self.memory.experiences)
        
        # Keep only experiences with significant triggers or state changes
        kept_experiences = []
        for exp in self.memory.experiences:
            # Keep if has trigger, or field value is notable
            if exp.trigger or abs(exp.field_value - 0.5) > threshold:
                kept_experiences.append(exp)
        
        # Update experiences
        self.memory.experiences.clear()
        self.memory.experiences.extend(kept_experiences[-self.memory.max_memory:])
        
        pruned = initial_count - len(self.memory.experiences)
        logger.info(f"Pruned {pruned} low-relevance memories")
        return pruned
    
    def consolidate_similar(self):
        """Merge similar experiences to reduce redundancy"""
        if len(self.memory.experiences) < 10:
            return 0
        
        # Group very similar consecutive experiences
        consolidated_count = 0
        experiences = list(self.memory.experiences)
        new_experiences = []
        
        i = 0
        while i < len(experiences):
            current = experiences[i]
            
            # Look for similar consecutive experiences
            similar_group = [current]
            j = i + 1
            while j < len(experiences) and j < i + 5:  # Check next 5 at most
                next_exp = experiences[j]
                # Similar if same context and close field values
                if (next_exp.context_state == current.context_state and 
                    abs(next_exp.field_value - current.field_value) < 0.1):
                    similar_group.append(next_exp)
                    j += 1
                else:
                    break
            
            if len(similar_group) > 2:
                # Create consolidated experience (keep first, note consolidation)
                consolidated = similar_group[0]
                consolidated.notes['consolidated_count'] = len(similar_group)
                new_experiences.append(consolidated)
                consolidated_count += len(similar_group) - 1
                i = j
            else:
                new_experiences.append(current)
                i += 1
        
        # Update experiences
        self.memory.experiences.clear()
        self.memory.experiences.extend(new_experiences)
        
        logger.info(f"Consolidated {consolidated_count} similar memories")
        return consolidated_count
    
    def prepare_for_wake(self):
        """Prepare memory systems for waking state"""
        logger.info("Preparing memory for wake state...")
        # Clear working memory and reload with most recent/relevant
        self.memory.working_memory.clear()
        
        # Load most recent experiences into working memory
        recent = list(self.memory.experiences)[-50:]
        for exp in recent:
            self.memory.working_memory.append(exp)
        
        # Update patterns one more time
        self.memory._update_patterns()
    
    def reinitialize_memory(self):
        """Reinitialize memory sensor with processed data"""
        logger.info("Reinitializing memory sensor...")
        # Trigger pattern updates
        self.memory._update_patterns()
        
        # Gradually restore trust in memory sensor
        memory_trust = self.engine.trust_model.base.get('memory', 0.5)
        self.engine.trust_model.base['memory'] = min(0.9, memory_trust + 0.1)
    
    def simulate_scenario(self, elements):
        """Run scenario through coherence engine"""
        # Simple simulation - in real implementation would be more sophisticated
        result = {
            'coherence': 0.5 + len(self.extracted_patterns) * 0.1,
            'elements_processed': len(elements)
        }
        return result
    
    def calculate_prediction_accuracies(self):
        """Calculate how well each sensor predicted outcomes"""
        accuracies = {}
        for sensor_id in self.engine.sensors:
            # Simple accuracy based on trust model
            avg_trust = sum(
                self.engine.trust_model.get(sensor_id, state) 
                for state in ContextState
            ) / len(ContextState)
            accuracies[sensor_id] = avg_trust
        return accuracies

class SleepScheduler:
    """Manages automatic sleep scheduling for the coherence engine"""
    
    def __init__(self, engine, memory_sensor, check_interval=60):
        self.engine = engine
        self.memory_sensor = memory_sensor
        self.sleep_cycle = IntegratedSleepCycle(engine, memory_sensor)
        self.monitor = SleepMonitor(self.sleep_cycle, check_interval)
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the sleep scheduler in background"""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("Sleep scheduler started")
        
    def _run(self):
        """Main scheduler loop"""
        while self.running:
            try:
                if not self.sleep_cycle.is_sleeping:
                    if self.sleep_cycle.should_sleep():
                        logger.info("Sleep cycle triggered")
                        self.sleep_cycle.enter_sleep()
                
                time.sleep(self.monitor.check_interval)
                
            except Exception as e:
                logger.error(f"Error in sleep scheduler: {e}")
                time.sleep(10)
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Sleep scheduler stopped")

def main():
    """Run coherence engine with integrated sleep cycles"""
    logger.info("Starting Coherence Engine with Sleep Cycles")
    
    # Initialize sensors
    memory_sensor = PersistentMemorySensor(memory_dir=Path("memory"))
    vision_sensor = RealVisionSensor()
    imu_sensor = RealIMUSensor()
    
    # Initialize coherence engine
    engine = CoherenceEngine(
        sensors=[memory_sensor, vision_sensor, imu_sensor],
        tick_rate=10  # 10 Hz
    )
    
    # Initialize sleep scheduler
    scheduler = SleepScheduler(
        engine=engine,
        memory_sensor=memory_sensor,
        check_interval=30  # Check every 30 seconds
    )
    
    # Handle shutdown
    def signal_handler(sig, frame):
        logger.info("\nShutting down...")
        scheduler.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start systems
    scheduler.start()
    
    # Run coherence engine
    logger.info("Running coherence engine with sleep cycles enabled")
    logger.info("Press Ctrl+C to stop")
    
    tick = 0
    while True:
        try:
            # Run engine step
            field = engine.step(tick)
            
            # Record experience in memory
            memory_sensor.observe(
                context_state=str(engine.context.state),
                sensor_readings={s.id: s.read(tick=tick) for s in engine.sensors},
                field_value=field,
                trigger=engine.context.last_trigger
            )
            
            # Display status
            if tick % 100 == 0:
                metrics = scheduler.sleep_cycle.get_metrics()
                urgency = metrics.sleep_urgency()
                logger.info(f"Tick {tick}: Field={field:.3f}, Sleep urgency={urgency:.2f}")
            
            tick += 1
            time.sleep(0.1)  # 10 Hz
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(1)
    
    # Cleanup
    scheduler.stop()
    logger.info("Coherence engine stopped")

if __name__ == "__main__":
    main()