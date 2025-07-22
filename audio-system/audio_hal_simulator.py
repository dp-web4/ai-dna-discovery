#!/usr/bin/env python3
"""
Audio HAL Simulator - For testing without audio hardware
Simulates audio input/output for development and testing
"""

import os
import numpy as np
import time
from typing import List, Optional, Tuple
from audio_hal import AudioBackend, AudioDevice, AudioHAL


class SimulatedAudioBackend(AudioBackend):
    """Simulated audio backend for testing"""
    
    def __init__(self):
        self.sim_time = 0
        self.noise_level = 0.01
        self.signal_generator = self._generate_test_signal
    
    def is_available(self) -> bool:
        return True
    
    def list_devices(self) -> List[AudioDevice]:
        """Return simulated devices"""
        devices = [
            AudioDevice(
                index=0,
                name="Simulated Microphone",
                channels_in=2,
                channels_out=0,
                sample_rate=16000,
                is_default=True
            ),
            AudioDevice(
                index=1,
                name="Simulated Speakers",
                channels_in=0,
                channels_out=2,
                sample_rate=16000,
                is_default=True
            ),
            AudioDevice(
                index=2,
                name="Simulated Headset",
                channels_in=1,
                channels_out=2,
                sample_rate=16000,
                is_default=False
            )
        ]
        
        for dev in devices:
            dev.backend = 'simulator'
        
        return devices
    
    def get_default_input_device(self) -> Optional[AudioDevice]:
        devices = self.list_devices()
        return devices[0]  # Simulated Microphone
    
    def get_default_output_device(self) -> Optional[AudioDevice]:
        devices = self.list_devices()
        return devices[1]  # Simulated Speakers
    
    def _generate_test_signal(self, duration: float, sample_rate: int) -> np.ndarray:
        """Generate simulated audio signal"""
        samples = int(duration * sample_rate)
        t = np.linspace(self.sim_time, self.sim_time + duration, samples)
        
        # Create interesting patterns
        signal = np.zeros(samples)
        
        # Add periodic "speech" bursts
        speech_period = 3.0  # seconds
        speech_duration = 0.5  # seconds
        
        for i in range(samples):
            time_pos = t[i] % speech_period
            if time_pos < speech_duration:
                # Simulate speech frequencies (100-400 Hz)
                signal[i] = (
                    0.3 * np.sin(2 * np.pi * 200 * t[i]) +
                    0.2 * np.sin(2 * np.pi * 150 * t[i]) +
                    0.1 * np.sin(2 * np.pi * 300 * t[i])
                )
                # Add envelope
                envelope = np.sin(np.pi * time_pos / speech_duration)
                signal[i] *= envelope
        
        # Add background noise
        noise = np.random.normal(0, self.noise_level, samples)
        signal += noise
        
        # Update simulation time
        self.sim_time += duration
        
        return signal
    
    def read_audio(self, device: AudioDevice, duration: float) -> np.ndarray:
        """Simulate reading audio"""
        # Simulate processing delay
        time.sleep(duration * 0.1)  # 10% of real time
        
        # Generate simulated audio
        audio = self._generate_test_signal(duration, device.sample_rate)
        
        # Add some randomness
        if np.random.random() < 0.1:  # 10% chance of loud event
            spike_pos = np.random.randint(0, len(audio))
            spike_width = int(0.05 * device.sample_rate)  # 50ms
            start = max(0, spike_pos - spike_width // 2)
            end = min(len(audio), spike_pos + spike_width // 2)
            audio[start:end] *= 5.0
        
        return audio
    
    def play_audio(self, device: AudioDevice, audio_data: np.ndarray, sample_rate: int):
        """Simulate playing audio"""
        duration = len(audio_data) / sample_rate
        print(f"[SIMULATOR] Playing {duration:.2f}s of audio to {device.name}")
        time.sleep(duration * 0.1)  # Simulate 10% of real time


class SimulatedTTS:
    """Simulated TTS for testing"""
    
    @staticmethod
    def speak(text: str, voice: str = "simulated", rate: int = 150, pitch: int = 50):
        """Simulate TTS output"""
        words = len(text.split())
        duration = words / (rate / 60.0)  # Approximate speaking time
        
        print(f"[TTS-SIM] Speaking ({voice}, rate={rate}, pitch={pitch}):")
        print(f"[TTS-SIM] \"{text}\"")
        print(f"[TTS-SIM] Duration: {duration:.1f}s")
        
        # Simulate speaking time
        time.sleep(duration * 0.2)  # 20% of real time


def create_simulated_hal(config_file: Optional[str] = None) -> 'AudioHAL':
    """Create HAL with simulated backend"""
    # Don't use AudioHAL constructor to avoid recursion
    import platform
    
    class SimulatedHAL:
        def __init__(self):
            self.config_file = config_file or "audio_config.json"
            self.config = {}
            self.platform = {
                'system': platform.system(),
                'machine': platform.machine(),
                'platform': platform.platform(),
                'processor': platform.processor(),
                'hostname': platform.node(),
                'python_version': platform.python_version(),
                'device_type': 'jetson' if os.path.exists('/etc/nv_tegra_release') else 'standard'
            }
            
            # Set up simulated backend
            sim_backend = SimulatedAudioBackend()
            self.backends = [sim_backend]
            self.active_backend = sim_backend
            self.input_device = sim_backend.get_default_input_device()
            self.output_device = sim_backend.get_default_output_device()
            
            # Platform-specific settings
            if self.platform['device_type'] == 'jetson':
                self.audio_settings = {
                    'gain': 50.0,
                    'threshold': 0.02,
                    'buffer_size': 1024
                }
            else:
                self.audio_settings = {
                    'gain': 1.0,
                    'threshold': 0.01,
                    'buffer_size': 512
                }
        
        def read_audio(self, duration: float):
            return self.active_backend.read_audio(self.input_device, duration), self.input_device.sample_rate
        
        def play_audio(self, audio_data, sample_rate):
            self.active_backend.play_audio(self.output_device, audio_data, sample_rate)
        
        def get_info(self):
            return {
                'platform': self.platform,
                'available_backends': ['simulator'],
                'active_backend': 'SimulatedAudioBackend',
                'input_device': str(self.input_device),
                'output_device': str(self.output_device),
                'audio_settings': self.audio_settings
            }
    
    return SimulatedHAL()


def test_simulator():
    """Test the audio simulator"""
    print("=== Audio HAL Simulator Test ===\n")
    
    # Create simulated HAL
    hal = create_simulated_hal()
    
    print("Platform:", hal.platform['device_type'])
    print("Simulated Devices:")
    for dev in hal.list_all_devices()['simulator']:
        print(f"  {dev}")
    
    print("\nTesting simulated audio input...")
    audio_data, sample_rate = hal.read_audio(2.0)
    
    print(f"Received {len(audio_data)} samples at {sample_rate}Hz")
    print(f"RMS level: {np.sqrt(np.mean(audio_data**2)):.3f}")
    print(f"Peak level: {np.max(np.abs(audio_data)):.3f}")
    
    print("\nTesting simulated audio output...")
    test_tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate))
    hal.play_audio(test_tone, sample_rate)
    
    print("\nTesting simulated TTS...")
    SimulatedTTS.speak("Hello! This is a simulated consciousness audio system!", rate=180)
    
    print("\n✅ Simulator working correctly!")


if __name__ == "__main__":
    test_simulator()