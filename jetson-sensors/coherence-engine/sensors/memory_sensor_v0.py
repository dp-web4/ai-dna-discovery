#!/usr/bin/env python3
"""
Memory Sensor - Temporal sensor that parses the past
"""

import json
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np
from collections import deque

from base_sensor import TemporalSensor, SensorReading

class MemorySensor(TemporalSensor):
    """Memory as a temporal sensor providing historical context"""
    
    def __init__(self, memory_path: str = "memory"):
        super().__init__("memory", "memory_sensor")
        self.memory_path = memory_path
        self.working_memory = deque(maxlen=100)  # Last 100 experiences
        self.pattern_cache = {}  # Cached patterns for quick access
        self.last_query_time = 0
        self.query_cache = {}  # Cache recent queries
        
    def initialize(self) -> bool:
        """Initialize memory sensor"""
        try:
            # Ensure memory directories exist
            os.makedirs(self.memory_path, exist_ok=True)
            
            # Load recent experiences into working memory
            self._load_recent_experiences()
            
            # Load known patterns
            self._load_patterns()
            
            return True
        except Exception as e:
            print(f"Memory sensor initialization error: {e}")
            return False
    
    def _load_recent_experiences(self):
        """Load recent experiences into working memory"""
        experiences_path = os.path.join(self.memory_path, "experiences")
        if not os.path.exists(experiences_path):
            return
        
        # Get last 3 days of experiences
        today = datetime.now()
        for days_back in range(3):
            date = today - timedelta(days=days_back)
            date_str = date.strftime("%Y-%m-%d")
            day_path = os.path.join(experiences_path, date_str)
            
            if os.path.exists(day_path):
                # Load experiences from this day
                for filename in sorted(os.listdir(day_path)):
                    if filename.endswith('.json'):
                        filepath = os.path.join(day_path, filename)
                        try:
                            with open(filepath, 'r') as f:
                                experience = json.load(f)
                                self.working_memory.append(experience)
                        except:
                            pass
    
    def _load_patterns(self):
        """Load recognized patterns"""
        patterns_path = os.path.join(self.memory_path, "patterns")
        if not os.path.exists(patterns_path):
            return
        
        # Load patterns from each category
        for category in ["spatial", "temporal", "contextual", "emergent"]:
            cat_path = os.path.join(patterns_path, category)
            if os.path.exists(cat_path):
                for filename in os.listdir(cat_path):
                    if filename.endswith('.json'):
                        pattern_id = filename[:-5]  # Remove .json
                        filepath = os.path.join(cat_path, filename)
                        try:
                            with open(filepath, 'r') as f:
                                pattern = json.load(f)
                                self.pattern_cache[pattern_id] = pattern
                        except:
                            pass
    
    def read(self) -> Optional[SensorReading]:
        """Get current memory reading based on context"""
        try:
            # Find relevant memories
            relevant_memories = self._find_relevant_memories()
            
            # Match patterns
            patterns_detected = self._detect_patterns(relevant_memories)
            
            # Generate prediction
            prediction = self._generate_prediction(relevant_memories, patterns_detected)
            
            # Calculate confidence based on memory match quality
            confidence = self._calculate_confidence(relevant_memories, patterns_detected)
            
            # Calculate relevance (how much history matters now)
            relevance = self._calculate_relevance()
            
            # Create reading
            reading = SensorReading(
                sensor_type="memory",
                timestamp=time.time(),
                data={
                    "relevant_memories": relevant_memories[:5],  # Top 5
                    "patterns_detected": patterns_detected,
                    "prediction": prediction,
                    "working_memory_size": len(self.working_memory),
                    "pattern_cache_size": len(self.pattern_cache)
                },
                confidence=confidence,
                relevance=relevance,
                metadata={
                    "query_time": time.time() - self.last_query_time
                }
            )
            
            self.last_query_time = time.time()
            return reading
            
        except Exception as e:
            print(f"Memory read error: {e}")
            return None
    
    def _find_relevant_memories(self) -> List[Dict]:
        """Find memories relevant to current context"""
        relevant = []
        
        # For now, return recent memories with similarity scores
        # In full implementation, would compare with current sensor state
        for memory in self.working_memory:
            # Simple recency-based relevance
            if 'timestamp' in memory:
                age = time.time() - memory['timestamp']
                recency_score = np.exp(-age / 3600)  # Decay over hours
                
                memory_with_score = memory.copy()
                memory_with_score['relevance_score'] = recency_score
                relevant.append(memory_with_score)
        
        # Sort by relevance
        relevant.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant
    
    def _detect_patterns(self, memories: List[Dict]) -> List[str]:
        """Detect known patterns in memories"""
        detected = []
        
        # Simple pattern detection - check if memory contexts match known patterns
        for pattern_id, pattern in self.pattern_cache.items():
            if self._matches_pattern(memories, pattern):
                detected.append(pattern_id)
        
        return detected
    
    def _matches_pattern(self, memories: List[Dict], pattern: Dict) -> bool:
        """Check if memories match a pattern"""
        # Simplified pattern matching
        if not memories:
            return False
        
        # Check if pattern conditions are met
        if 'context_state' in pattern:
            # Check if any memory has matching context
            for memory in memories[:5]:  # Check top 5
                if memory.get('context', {}).get('state') == pattern['context_state']:
                    return True
        
        return False
    
    def _generate_prediction(self, memories: List[Dict], patterns: List[str]) -> Dict:
        """Generate prediction based on memories and patterns"""
        prediction = {
            "likely_next_state": "stable",
            "confidence": 0.5,
            "based_on": []
        }
        
        if memories:
            # Look at state transitions in memories
            states = []
            for memory in memories[:10]:
                if 'context' in memory and 'state' in memory['context']:
                    states.append(memory['context']['state'])
            
            if states:
                # Predict most common next state (simplified)
                from collections import Counter
                state_counts = Counter(states)
                prediction["likely_next_state"] = state_counts.most_common(1)[0][0]
                prediction["confidence"] = state_counts.most_common(1)[0][1] / len(states)
                prediction["based_on"] = patterns[:3] if patterns else ["recent_history"]
        
        return prediction
    
    def _calculate_confidence(self, memories: List[Dict], patterns: List[str]) -> float:
        """Calculate confidence in memory reading"""
        if not memories:
            return 0.1  # Low confidence with no memories
        
        # Base confidence on memory quality and pattern matches
        memory_score = min(1.0, len(memories) / 20)  # More memories = higher confidence
        pattern_score = min(1.0, len(patterns) / 3)  # More patterns = higher confidence
        
        # Weight average
        confidence = (memory_score * 0.6 + pattern_score * 0.4)
        
        return confidence
    
    def _calculate_relevance(self) -> float:
        """Calculate how relevant memory is to current situation"""
        # In stable situations, memory is less relevant
        # In novel situations, memory is more relevant for comparison
        
        # Simple implementation - would check actual context
        base_relevance = 0.5
        
        # Adjust based on working memory size
        if len(self.working_memory) > 50:
            base_relevance += 0.2  # Rich history available
        elif len(self.working_memory) < 10:
            base_relevance -= 0.2  # Limited history
        
        return max(0.1, min(1.0, base_relevance))
    
    def get_temporal_context(self) -> Dict[str, Any]:
        """Get temporal context from memory"""
        return {
            "working_memory_size": len(self.working_memory),
            "patterns_known": len(self.pattern_cache),
            "oldest_memory": self.working_memory[0]['timestamp'] if self.working_memory else None,
            "newest_memory": self.working_memory[-1]['timestamp'] if self.working_memory else None
        }
    
    def predict_future(self, timesteps: int) -> Dict[str, Any]:
        """Predict future states based on memory"""
        # Simple prediction based on recent patterns
        predictions = []
        
        # Look at recent state transitions
        recent_states = []
        for memory in list(self.working_memory)[-20:]:
            if 'context' in memory and 'state' in memory['context']:
                recent_states.append(memory['context']['state'])
        
        if recent_states:
            # Simple markov-like prediction
            for i in range(timesteps):
                # Predict based on most common transition
                if len(recent_states) > i:
                    predictions.append({
                        "timestep": i + 1,
                        "predicted_state": recent_states[-(i+1)],  # Cycle through recent
                        "confidence": 0.7 - (i * 0.1)  # Decay confidence
                    })
        
        return {
            "predictions": predictions,
            "based_on_memories": len(recent_states)
        }
    
    def store_experience(self, experience: Dict):
        """Store a new experience in memory"""
        # Add to working memory
        self.working_memory.append(experience)
        
        # Also save to disk if significant
        if experience.get('attention_triggers'):
            self._save_significant_experience(experience)
    
    def _save_significant_experience(self, experience: Dict):
        """Save significant experience to disk"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        day_path = os.path.join(self.memory_path, "experiences", date_str)
        os.makedirs(day_path, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%H-%M-%S-%f")[:-3]
        filename = os.path.join(day_path, f"{timestamp_str}.json")
        
        with open(filename, 'w') as f:
            json.dump(experience, f, indent=2)
    
    def learn_pattern(self, pattern_id: str, pattern_data: Dict, category: str = "emergent"):
        """Learn a new pattern"""
        # Store in cache
        self.pattern_cache[pattern_id] = pattern_data
        
        # Save to disk
        cat_path = os.path.join(self.memory_path, "patterns", category)
        os.makedirs(cat_path, exist_ok=True)
        
        filename = os.path.join(cat_path, f"{pattern_id}.json")
        with open(filename, 'w') as f:
            json.dump(pattern_data, f, indent=2)
    
    def calibrate(self) -> bool:
        """Calibrate memory sensor"""
        # Memory doesn't need traditional calibration
        # But we can optimize caches
        
        # Trim working memory if too large
        if len(self.working_memory) > 100:
            # Keep only recent
            recent = list(self.working_memory)[-100:]
            self.working_memory.clear()
            self.working_memory.extend(recent)
        
        # Clear query cache
        self.query_cache.clear()
        
        return True