"""
Persistent memory sensor with file-based storage.
Implements temporal parsing of past experiences.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np
import logging

logger = logging.getLogger("persistent_memory")

@dataclass
class Experience:
    """Single experience snapshot."""
    timestamp: float
    context_state: str
    sensor_readings: Dict[str, float]
    field_value: float
    trigger: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class PersistentMemorySensor:
    """
    Memory sensor with persistent storage and pattern recognition.
    Treats memory as active temporal sensor parsing the past.
    """
    id: str = "memory"
    memory_dir: Path = field(default_factory=lambda: Path("memory"))
    recency_bias: float = 0.6  # Weight recent patterns more
    pattern_window: int = 20  # Look for patterns in last N experiences
    max_memory: int = 10000  # Maximum experiences to keep in memory
    
    # Runtime state
    experiences: deque = field(default_factory=lambda: deque(maxlen=10000), init=False)
    working_memory: deque = field(default_factory=lambda: deque(maxlen=100), init=False)
    patterns: Dict[str, float] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize memory system and load existing memories."""
        # Create directory structure
        self.experiences_dir = self.memory_dir / "experiences"
        self.patterns_dir = self.memory_dir / "patterns"
        self.context_dir = self.memory_dir / "context"
        
        for dir in [self.experiences_dir, self.patterns_dir, self.context_dir]:
            dir.mkdir(parents=True, exist_ok=True)
            
        # Load existing experiences
        self._load_experiences()
        
        # Initialize pattern detection
        self._update_patterns()
        
        logger.info(f"Memory sensor initialized with {len(self.experiences)} experiences")
        
    def _load_experiences(self):
        """Load experiences from disk."""
        try:
            # Load most recent experience file
            experience_files = sorted(self.experiences_dir.glob("*.json"))
            if experience_files:
                latest_file = experience_files[-1]
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    for exp_data in data[-self.max_memory:]:  # Load last N
                        exp = Experience(**exp_data)
                        self.experiences.append(exp)
                        self.working_memory.append(exp)
                logger.info(f"Loaded {len(self.experiences)} experiences from {latest_file}")
        except Exception as e:
            logger.warning(f"Could not load experiences: {e}")
            
    def _save_experience(self, experience: Experience):
        """Save experience to disk."""
        try:
            # Save to daily file
            date_str = time.strftime("%Y%m%d")
            file_path = self.experiences_dir / f"experiences_{date_str}.json"
            
            # Load existing or create new
            existing = []
            if file_path.exists():
                with open(file_path, 'r') as f:
                    existing = json.load(f)
                    
            # Append new experience
            existing.append(asdict(experience))
            
            # Save back
            with open(file_path, 'w') as f:
                json.dump(existing, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save experience: {e}")
            
    def observe(self, context_state: str, sensor_readings: Dict[str, float], 
                field_value: float, trigger: Optional[str] = None):
        """
        Record a new experience.
        Called by coherence engine after each step.
        """
        experience = Experience(
            timestamp=time.time(),
            context_state=context_state,
            sensor_readings=sensor_readings,
            field_value=field_value,
            trigger=trigger
        )
        
        self.experiences.append(experience)
        self.working_memory.append(experience)
        self._save_experience(experience)
        self._update_patterns()
        
    def _update_patterns(self):
        """Detect patterns in recent experiences."""
        if len(self.working_memory) < 3:
            return
            
        # Pattern: Context state transitions
        states = [exp.context_state for exp in self.working_memory]
        state_transitions = {}
        for i in range(len(states) - 1):
            key = f"{states[i]}->{states[i+1]}"
            state_transitions[key] = state_transitions.get(key, 0) + 1
            
        # Normalize to probabilities
        total = sum(state_transitions.values())
        if total > 0:
            for key in state_transitions:
                self.patterns[f"transition_{key}"] = state_transitions[key] / total
                
        # Pattern: Average field stability
        if len(self.working_memory) >= 5:
            recent_fields = [exp.field_value for exp in list(self.working_memory)[-5:]]
            stability = 1.0 / (1.0 + np.std(recent_fields))
            self.patterns["field_stability"] = stability
            
        # Pattern: Trigger frequency
        recent_triggers = [exp.trigger for exp in list(self.working_memory)[-20:] 
                          if exp.trigger]
        if recent_triggers:
            trigger_rate = len(recent_triggers) / min(20, len(self.working_memory))
            self.patterns["trigger_rate"] = trigger_rate
            
    def _find_similar_experiences(self, current_context: str, 
                                 current_readings: Dict[str, float]) -> List[Experience]:
        """Find past experiences similar to current situation."""
        similar = []
        
        for exp in self.experiences:
            # Context match
            if exp.context_state == current_context:
                similarity = 1.0
            else:
                similarity = 0.5
                
            # Sensor reading similarity (if available)
            if current_readings and exp.sensor_readings:
                common_sensors = set(current_readings.keys()) & set(exp.sensor_readings.keys())
                if common_sensors:
                    diffs = [abs(current_readings[s] - exp.sensor_readings[s]) 
                            for s in common_sensors]
                    reading_similarity = 1.0 / (1.0 + np.mean(diffs))
                    similarity *= reading_similarity
                    
            if similarity > 0.3:  # Threshold for "similar"
                similar.append((similarity, exp))
                
        # Sort by similarity and return top matches
        similar.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in similar[:10]]
        
    def read(self, *, tick: int) -> float:
        """
        Return memory confidence based on pattern recognition and stability.
        High values indicate familiar/stable patterns, low values indicate novelty.
        """
        if len(self.experiences) < 5:
            # Not enough history
            return 0.2
            
        # Base confidence from field stability
        stability = self.patterns.get("field_stability", 0.5)
        
        # Adjust based on trigger rate (high triggers = less stable)
        trigger_rate = self.patterns.get("trigger_rate", 0.0)
        stability *= (1.0 - trigger_rate * 0.5)
        
        # Recency weighting
        if self.working_memory:
            # Recent average field value
            recent_fields = [exp.field_value for exp in list(self.working_memory)[-10:]]
            recent_avg = np.mean(recent_fields)
            
            # Long-term average
            all_fields = [exp.field_value for exp in list(self.experiences)[-100:]]
            long_avg = np.mean(all_fields)
            
            # Deviation from long-term patterns
            deviation = abs(recent_avg - long_avg)
            familiarity = 1.0 / (1.0 + deviation * 2)
            
            # Combine with recency bias
            confidence = (self.recency_bias * stability + 
                         (1 - self.recency_bias) * familiarity)
        else:
            confidence = stability
            
        return min(1.0, max(0.0, confidence))
        
    def get_insights(self) -> Dict[str, Any]:
        """Get memory insights for debugging/visualization."""
        return {
            "total_experiences": len(self.experiences),
            "working_memory_size": len(self.working_memory),
            "patterns": self.patterns.copy(),
            "recent_contexts": [exp.context_state for exp in list(self.working_memory)[-5:]],
            "stability": self.patterns.get("field_stability", 0.0),
            "trigger_rate": self.patterns.get("trigger_rate", 0.0)
        }