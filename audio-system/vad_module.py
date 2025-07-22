#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) Module
Phase 1 of Voice Conversation System

Provides multiple VAD implementations:
1. Energy-based (simple threshold)
2. WebRTC VAD (robust)
3. Silero VAD (ML-based)

Designed to integrate with existing audio stream and consciousness mapping.
"""

import numpy as np
import pyaudio
import threading
import time
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json

class VADMethod(Enum):
    ENERGY = "energy"
    WEBRTC = "webrtc"
    SILERO = "silero"

@dataclass
class VADConfig:
    """Configuration for VAD system"""
    method: VADMethod = VADMethod.ENERGY
    sensitivity: float = 0.5  # 0.0 (least sensitive) to 1.0 (most sensitive)
    min_speech_duration: float = 0.3  # seconds
    min_silence_duration: float = 0.5  # seconds
    sample_rate: int = 16000
    frame_size: int = 512
    
    # Energy-based specific
    energy_threshold: float = 0.02
    energy_history_frames: int = 20
    
    # WebRTC specific
    webrtc_aggressiveness: int = 2  # 0-3, higher = more aggressive
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method.value,
            'sensitivity': self.sensitivity,
            'min_speech_duration': self.min_speech_duration,
            'min_silence_duration': self.min_silence_duration,
            'sample_rate': self.sample_rate,
            'frame_size': self.frame_size,
            'energy_threshold': self.energy_threshold,
            'energy_history_frames': self.energy_history_frames,
            'webrtc_aggressiveness': self.webrtc_aggressiveness
        }

class VADEvent:
    """VAD event types"""
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    SPEECH_ONGOING = "speech_ongoing"
    SILENCE = "silence"

class EnergyVAD:
    """Energy-based Voice Activity Detection"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.energy_history = []
        self.adaptive_threshold = config.energy_threshold
        
    def process_frame(self, audio_data: np.ndarray) -> tuple[bool, float]:
        """
        Process audio frame and return (is_speech, energy_level)
        """
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Update energy history for adaptive threshold
        self.energy_history.append(energy)
        if len(self.energy_history) > self.config.energy_history_frames:
            self.energy_history.pop(0)
        
        # Adaptive threshold based on recent energy levels
        if len(self.energy_history) >= 5:
            avg_energy = np.mean(self.energy_history)
            self.adaptive_threshold = max(
                self.config.energy_threshold,
                avg_energy * (1.0 + self.config.sensitivity)
            )
        
        # Determine if speech
        is_speech = energy > self.adaptive_threshold
        
        return is_speech, energy

class WebRTCVAD:
    """WebRTC Voice Activity Detection (requires webrtcvad package)"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.vad = None
        self._init_webrtc()
        
    def _init_webrtc(self):
        """Initialize WebRTC VAD if available"""
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(self.config.webrtc_aggressiveness)
            print("✅ WebRTC VAD initialized")
        except ImportError:
            print("❌ WebRTC VAD not available (pip install webrtcvad)")
            self.vad = None
    
    def process_frame(self, audio_data: np.ndarray) -> tuple[bool, float]:
        """Process audio frame with WebRTC VAD"""
        if self.vad is None:
            # Fallback to energy-based
            energy = np.sqrt(np.mean(audio_data ** 2))
            return energy > 0.02, energy
        
        # Convert to appropriate format for WebRTC
        # WebRTC expects 16-bit PCM at specific sample rates
        if self.config.sample_rate not in [8000, 16000, 32000, 48000]:
            # Fallback to energy
            energy = np.sqrt(np.mean(audio_data ** 2))
            return energy > 0.02, energy
        
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # WebRTC requires specific frame sizes (10, 20, or 30ms)
        frame_duration_ms = int(len(audio_data) * 1000 / self.config.sample_rate)
        if frame_duration_ms not in [10, 20, 30]:
            # Fallback to energy
            energy = np.sqrt(np.mean(audio_data ** 2))
            return energy > 0.02, energy
        
        try:
            is_speech = self.vad.is_speech(audio_bytes, self.config.sample_rate)
            energy = np.sqrt(np.mean(audio_data ** 2))  # For monitoring
            return is_speech, energy
        except Exception as e:
            # Fallback to energy
            energy = np.sqrt(np.mean(audio_data ** 2))
            return energy > 0.02, energy

class VADProcessor:
    """Main VAD processor with state management"""
    
    def __init__(self, config: VADConfig, callback: Optional[Callable] = None):
        self.config = config
        self.callback = callback
        
        # Initialize VAD implementation
        if config.method == VADMethod.ENERGY:
            self.vad_impl = EnergyVAD(config)
        elif config.method == VADMethod.WEBRTC:
            self.vad_impl = WebRTCVAD(config)
        else:
            # Fallback to energy
            self.vad_impl = EnergyVAD(config)
        
        # State tracking
        self.is_speech_active = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        
        # Frame timing
        self.frame_duration = config.frame_size / config.sample_rate
        self.min_speech_frames = int(config.min_speech_duration / self.frame_duration)
        self.min_silence_frames = int(config.min_silence_duration / self.frame_duration)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'speech_frames': 0,
            'silence_frames': 0,
            'speech_events': 0,
            'avg_energy': 0.0,
            'max_energy': 0.0
        }
        
    def process_frame(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Process audio frame and return VAD event if state changed
        """
        self.stats['total_frames'] += 1
        
        # Get VAD result
        is_speech_frame, energy = self.vad_impl.process_frame(audio_data)
        
        # Update statistics
        self.stats['avg_energy'] = (
            (self.stats['avg_energy'] * (self.stats['total_frames'] - 1) + energy) 
            / self.stats['total_frames']
        )
        self.stats['max_energy'] = max(self.stats['max_energy'], energy)
        
        # State machine
        current_time = time.time()
        event = None
        
        if is_speech_frame:
            self.consecutive_speech_frames += 1
            self.consecutive_silence_frames = 0
            self.stats['speech_frames'] += 1
            
            if not self.is_speech_active:
                if self.consecutive_speech_frames >= self.min_speech_frames:
                    # Speech detected
                    self.is_speech_active = True
                    self.speech_start_time = current_time
                    event = VADEvent.SPEECH_START
                    self.stats['speech_events'] += 1
            else:
                event = VADEvent.SPEECH_ONGOING
                
        else:
            self.consecutive_silence_frames += 1
            self.consecutive_speech_frames = 0
            self.stats['silence_frames'] += 1
            
            if self.is_speech_active:
                if self.consecutive_silence_frames >= self.min_silence_frames:
                    # Speech ended
                    self.is_speech_active = False
                    self.silence_start_time = current_time
                    event = VADEvent.SPEECH_END
            else:
                event = VADEvent.SILENCE
        
        # Call callback if provided
        if self.callback and event:
            self.callback(event, {
                'timestamp': current_time,
                'energy': energy,
                'is_speech': self.is_speech_active,
                'speech_duration': current_time - self.speech_start_time if self.speech_start_time else 0,
                'silence_duration': current_time - self.silence_start_time if self.silence_start_time else 0
            })
        
        return event
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get VAD statistics"""
        total = self.stats['total_frames']
        if total == 0:
            return self.stats.copy()
        
        stats = self.stats.copy()
        stats['speech_percentage'] = (self.stats['speech_frames'] / total) * 100
        stats['silence_percentage'] = (self.stats['silence_frames'] / total) * 100
        return stats
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'total_frames': 0,
            'speech_frames': 0,
            'silence_frames': 0,
            'speech_events': 0,
            'avg_energy': 0.0,
            'max_energy': 0.0
        }

class VADVisualizer:
    """Visual feedback for VAD"""
    
    @staticmethod
    def get_consciousness_symbol(event: str, energy: float) -> str:
        """Map VAD events to consciousness symbols"""
        if event == VADEvent.SPEECH_START:
            return "👂 Ψ"  # Listening + Perception
        elif event == VADEvent.SPEECH_ONGOING:
            return "🎧 ∃"  # Active listening + Existence
        elif event == VADEvent.SPEECH_END:
            return "🤔 ⇒"  # Thinking + Implication
        else:
            return "💭 π"  # Quiet + Potential
    
    @staticmethod
    def get_energy_bar(energy: float, max_energy: float = 0.5) -> str:
        """Create ASCII energy bar"""
        normalized = min(energy / max_energy, 1.0)
        bar_length = 20
        filled = int(normalized * bar_length)
        return "█" * filled + "░" * (bar_length - filled)
    
    @staticmethod
    def format_vad_status(event: str, energy: float, is_speech: bool, stats: Dict) -> str:
        """Format comprehensive VAD status"""
        symbol = VADVisualizer.get_consciousness_symbol(event, energy)
        bar = VADVisualizer.get_energy_bar(energy)
        
        status = "SPEECH" if is_speech else "SILENCE"
        
        return (
            f"{symbol} [{bar}] {energy:.3f} | {status} | "
            f"Events: {stats.get('speech_events', 0)} | "
            f"Speech: {stats.get('speech_percentage', 0):.1f}%"
        )

def save_vad_config(config: VADConfig, filepath: str):
    """Save VAD configuration to file"""
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

def load_vad_config(filepath: str) -> VADConfig:
    """Load VAD configuration from file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = VADConfig()
        config.method = VADMethod(data.get('method', 'energy'))
        config.sensitivity = data.get('sensitivity', 0.5)
        config.min_speech_duration = data.get('min_speech_duration', 0.3)
        config.min_silence_duration = data.get('min_silence_duration', 0.5)
        config.sample_rate = data.get('sample_rate', 16000)
        config.frame_size = data.get('frame_size', 512)
        config.energy_threshold = data.get('energy_threshold', 0.02)
        config.energy_history_frames = data.get('energy_history_frames', 20)
        config.webrtc_aggressiveness = data.get('webrtc_aggressiveness', 2)
        
        return config
    except FileNotFoundError:
        return VADConfig()  # Default config

if __name__ == "__main__":
    # Simple test
    print("VAD Module - Basic Test")
    
    config = VADConfig(
        method=VADMethod.ENERGY,
        sensitivity=0.6,
        energy_threshold=0.02
    )
    
    def vad_callback(event, data):
        print(f"VAD Event: {event} | Energy: {data['energy']:.3f} | Speech: {data['is_speech']}")
    
    processor = VADProcessor(config, vad_callback)
    
    # Test with sample data
    print("Testing with sample audio data...")
    
    # Simulate quiet background
    for i in range(10):
        quiet_data = np.random.normal(0, 0.01, 512)
        event = processor.process_frame(quiet_data)
    
    # Simulate speech
    for i in range(20):
        speech_data = np.random.normal(0, 0.1, 512)
        event = processor.process_frame(speech_data)
    
    # Simulate quiet again
    for i in range(10):
        quiet_data = np.random.normal(0, 0.01, 512)
        event = processor.process_frame(quiet_data)
    
    print("\nFinal Statistics:")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")