#!/usr/bin/env python3
"""
Logging Effector Plugin
Records system state as a form of persistent memory
August 12, 2025
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from collections import deque

class LoggingEffector:
    """
    Logging as an effector - writing memory to persistent storage.
    Can be configured to log different aspects at different rates.
    """
    
    def __init__(self, 
                 log_dir: str = "experiments/trust-dynamics",
                 log_rate: float = 10.0,  # Hz
                 buffer_size: int = 100):
        """
        Initialize logging effector
        
        Args:
            log_dir: Directory for log files
            log_rate: Logging frequency in Hz
            buffer_size: Buffer size before flush to disk
        """
        self.id = "logging_effector"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging configuration
        self.log_rate = log_rate
        self.log_interval = 1.0 / log_rate
        self.last_log_time = 0
        
        # Buffer for batch writes (efficiency)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        
        # Current log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"trust_log_{timestamp}.jsonl"
        
        # Experiment phases for annotation
        self.current_phase = "baseline"
        self.phase_start_time = time.time()
        
        # Statistics tracking
        self.stats = {
            "entries_logged": 0,
            "flushes": 0,
            "start_time": time.time(),
            "phases": []
        }
        
    def set_phase(self, phase: str):
        """Mark experiment phase transition"""
        self.stats["phases"].append({
            "phase": self.current_phase,
            "duration": time.time() - self.phase_start_time
        })
        self.current_phase = phase
        self.phase_start_time = time.time()
        print(f"[LoggingEffector] Phase transition: {phase}")
        
    def should_log(self) -> bool:
        """Check if it's time to log based on rate limit"""
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            return True
        return False
        
    def effect(self, reality_field: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Effect: Write system state to persistent memory
        
        Args:
            reality_field: Current reality field value
            context: Full context including sensors, trust, state
            
        Returns:
            Status of logging operation
        """
        if not self.should_log():
            return {"logged": False, "reason": "rate_limit"}
            
        # Extract relevant data
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tick": context.get("tick", 0),
            "phase": self.current_phase,
            "reality_field": reality_field,
            "context_state": context.get("state", "UNKNOWN"),
            "trust_weights": self._extract_trust_weights(context),
            "sensor_readings": self._extract_sensor_data(context),
            "conflict_detected": self._detect_conflict(context),
            "metadata": {
                "phase_time": time.time() - self.phase_start_time,
                "total_time": time.time() - self.stats["start_time"]
            }
        }
        
        # Add to buffer
        self.buffer.append(log_entry)
        self.stats["entries_logged"] += 1
        
        # Flush if buffer full
        if len(self.buffer) >= self.buffer_size:
            self.flush()
            
        return {
            "logged": True,
            "entry_count": self.stats["entries_logged"],
            "buffer_size": len(self.buffer)
        }
        
    def _extract_trust_weights(self, context: Dict) -> Dict[str, float]:
        """Extract trust weights for each sensor"""
        trust = {}
        
        # Handle different trust weight formats
        if "trust_weights" in context:
            weights = context["trust_weights"]
            if isinstance(weights, dict):
                # Separate camera weights if available
                if "camera" in weights:
                    trust["camera_combined"] = weights["camera"]
                if "camera_left" in weights:
                    trust["camera_left"] = weights["camera_left"]
                if "camera_right" in weights:
                    trust["camera_right"] = weights["camera_right"]
                if "imu" in weights:
                    trust["imu"] = weights["imu"]
            else:
                # Legacy format
                trust["unknown"] = 0.5
                
        return trust
        
    def _extract_sensor_data(self, context: Dict) -> Dict[str, Any]:
        """Extract sensor readings"""
        sensors = {}
        
        # Camera motion
        if "camera_motion" in context:
            sensors["camera_motion"] = context["camera_motion"]
            
        # IMU stability
        if "imu_stability" in context:
            sensors["imu_stability"] = context["imu_stability"]
            
        # IMU raw data if available
        if "imu_data" in context:
            imu = context["imu_data"]
            if "gyroscope" in imu:
                import numpy as np
                sensors["gyro_magnitude"] = float(np.linalg.norm(imu["gyroscope"]))
            if "acceleration" in imu:
                sensors["accel_magnitude"] = float(np.linalg.norm(imu["acceleration"]))
                
        return sensors
        
    def _detect_conflict(self, context: Dict) -> bool:
        """Detect if sensors are in conflict"""
        # Simple heuristic: high motion but high stability = conflict
        motion = context.get("camera_motion", 0)
        stability = context.get("imu_stability", 1)
        
        # Conflict if camera sees motion but IMU is stable, or vice versa
        if (motion > 0.7 and stability > 0.7) or (motion < 0.3 and stability < 0.3):
            return True
            
        return False
        
    def flush(self):
        """Flush buffer to disk"""
        if not self.buffer:
            return
            
        # Write all buffered entries
        with open(self.log_file, 'a') as f:
            for entry in self.buffer:
                f.write(json.dumps(entry) + '\n')
                
        self.stats["flushes"] += 1
        self.buffer.clear()
        
    def finalize(self) -> Dict[str, Any]:
        """Finalize logging and write statistics"""
        # Flush remaining buffer
        self.flush()
        
        # Add final phase
        if self.current_phase:
            self.stats["phases"].append({
                "phase": self.current_phase,
                "duration": time.time() - self.phase_start_time
            })
            
        # Write statistics file
        stats_file = self.log_file.with_suffix('.stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
        print(f"[LoggingEffector] Finalized: {self.stats['entries_logged']} entries")
        print(f"[LoggingEffector] Log file: {self.log_file}")
        print(f"[LoggingEffector] Stats file: {stats_file}")
        
        return self.stats
        
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze logged data for trust dynamics patterns
        Returns summary statistics
        """
        if not self.log_file.exists():
            return {"error": "No log file found"}
            
        # Read all entries
        entries = []
        with open(self.log_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line))
                
        if not entries:
            return {"error": "No entries found"}
            
        # Analyze trust evolution
        analysis = {
            "total_entries": len(entries),
            "duration": entries[-1]["metadata"]["total_time"],
            "phases": {},
            "trust_changes": {},
            "conflicts": 0,
            "context_transitions": []
        }
        
        # Group by phase
        for phase in set(e["phase"] for e in entries):
            phase_entries = [e for e in entries if e["phase"] == phase]
            if phase_entries:
                analysis["phases"][phase] = {
                    "count": len(phase_entries),
                    "avg_reality_field": sum(e["reality_field"] for e in phase_entries) / len(phase_entries),
                    "trust_range": self._calculate_trust_range(phase_entries)
                }
                
        # Calculate trust weight changes
        if len(entries) > 1:
            first_trust = entries[0]["trust_weights"]
            last_trust = entries[-1]["trust_weights"]
            for sensor in first_trust:
                if sensor in last_trust:
                    analysis["trust_changes"][sensor] = {
                        "initial": first_trust[sensor],
                        "final": last_trust[sensor],
                        "change": last_trust[sensor] - first_trust[sensor]
                    }
                    
        # Count conflicts
        analysis["conflicts"] = sum(1 for e in entries if e.get("conflict_detected", False))
        
        # Track context transitions
        last_state = None
        for e in entries:
            state = e.get("context_state")
            if state and state != last_state:
                analysis["context_transitions"].append({
                    "from": last_state,
                    "to": state,
                    "time": e["metadata"]["total_time"]
                })
                last_state = state
                
        return analysis
        
    def _calculate_trust_range(self, entries: list) -> Dict:
        """Calculate min/max trust values in a set of entries"""
        trust_ranges = {}
        
        for entry in entries:
            for sensor, value in entry.get("trust_weights", {}).items():
                if sensor not in trust_ranges:
                    trust_ranges[sensor] = {"min": value, "max": value}
                else:
                    trust_ranges[sensor]["min"] = min(trust_ranges[sensor]["min"], value)
                    trust_ranges[sensor]["max"] = max(trust_ranges[sensor]["max"], value)
                    
        return trust_ranges


# Factory function for plugin system
def create_effector(**kwargs):
    """Create logging effector instance"""
    return LoggingEffector(**kwargs)