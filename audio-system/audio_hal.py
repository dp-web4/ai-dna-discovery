#!/usr/bin/env python3
"""
Audio Hardware Abstraction Layer (HAL)
Provides platform-independent audio interface with automatic hardware detection
"""

import os
import sys
import platform
import subprocess
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple
import numpy as np

# Try importing audio libraries
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class AudioDevice:
    """Represents an audio device with its properties"""
    def __init__(self, index: int, name: str, channels_in: int, channels_out: int, 
                 sample_rate: int = 16000, is_default: bool = False):
        self.index = index
        self.name = name
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.sample_rate = sample_rate
        self.is_default = is_default
        self.backend = None
        
    def __repr__(self):
        return f"AudioDevice({self.index}: {self.name}, in={self.channels_in}, out={self.channels_out})"


class AudioBackend(ABC):
    """Abstract base class for audio backends"""
    
    @abstractmethod
    def list_devices(self) -> List[AudioDevice]:
        """List all available audio devices"""
        pass
    
    @abstractmethod
    def get_default_input_device(self) -> Optional[AudioDevice]:
        """Get the default input device"""
        pass
    
    @abstractmethod
    def get_default_output_device(self) -> Optional[AudioDevice]:
        """Get the default output device"""
        pass
    
    @abstractmethod
    def read_audio(self, device: AudioDevice, duration: float) -> np.ndarray:
        """Read audio from device for specified duration"""
        pass
    
    @abstractmethod
    def play_audio(self, device: AudioDevice, audio_data: np.ndarray, sample_rate: int):
        """Play audio data to device"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the system"""
        pass


class PyAudioBackend(AudioBackend):
    """PyAudio backend implementation"""
    
    def __init__(self):
        self.pa = None
        if HAS_PYAUDIO:
            try:
                self.pa = pyaudio.PyAudio()
            except:
                self.pa = None
    
    def is_available(self) -> bool:
        return self.pa is not None
    
    def list_devices(self) -> List[AudioDevice]:
        if not self.is_available():
            return []
        
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            device = AudioDevice(
                index=i,
                name=info['name'],
                channels_in=info['maxInputChannels'],
                channels_out=info['maxOutputChannels'],
                sample_rate=int(info['defaultSampleRate']),
                is_default=(i == self.pa.get_default_input_device_info()['index'] or 
                           i == self.pa.get_default_output_device_info()['index'])
            )
            device.backend = 'pyaudio'
            devices.append(device)
        return devices
    
    def get_default_input_device(self) -> Optional[AudioDevice]:
        if not self.is_available():
            return None
        
        try:
            info = self.pa.get_default_input_device_info()
            return AudioDevice(
                index=info['index'],
                name=info['name'],
                channels_in=info['maxInputChannels'],
                channels_out=info['maxOutputChannels'],
                sample_rate=int(info['defaultSampleRate']),
                is_default=True
            )
        except:
            return None
    
    def get_default_output_device(self) -> Optional[AudioDevice]:
        if not self.is_available():
            return None
        
        try:
            info = self.pa.get_default_output_device_info()
            return AudioDevice(
                index=info['index'],
                name=info['name'],
                channels_in=info['maxInputChannels'],
                channels_out=info['maxOutputChannels'],
                sample_rate=int(info['defaultSampleRate']),
                is_default=True
            )
        except:
            return None
    
    def read_audio(self, device: AudioDevice, duration: float) -> np.ndarray:
        if not self.is_available():
            return np.array([])
        
        stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=min(device.channels_in, 2),
            rate=device.sample_rate,
            input=True,
            input_device_index=device.index,
            frames_per_buffer=1024
        )
        
        frames = []
        for _ in range(int(device.sample_rate * duration / 1024)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.float32))
        
        stream.stop_stream()
        stream.close()
        
        return np.concatenate(frames)
    
    def play_audio(self, device: AudioDevice, audio_data: np.ndarray, sample_rate: int):
        if not self.is_available():
            return
        
        stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=min(device.channels_out, 2),
            rate=sample_rate,
            output=True,
            output_device_index=device.index
        )
        
        stream.write(audio_data.astype(np.float32).tobytes())
        stream.stop_stream()
        stream.close()


class SoundDeviceBackend(AudioBackend):
    """Sounddevice backend implementation"""
    
    def is_available(self) -> bool:
        return HAS_SOUNDDEVICE
    
    def list_devices(self) -> List[AudioDevice]:
        if not self.is_available():
            return []
        
        devices = []
        device_list = sd.query_devices()
        
        for i, dev in enumerate(device_list):
            device = AudioDevice(
                index=i,
                name=dev['name'],
                channels_in=dev['max_input_channels'],
                channels_out=dev['max_output_channels'],
                sample_rate=int(dev['default_samplerate']),
                is_default=(i == sd.default.device[0] or i == sd.default.device[1])
            )
            device.backend = 'sounddevice'
            devices.append(device)
        
        return devices
    
    def get_default_input_device(self) -> Optional[AudioDevice]:
        if not self.is_available():
            return None
        
        try:
            idx = sd.default.device[0]
            if idx is None:
                return None
            dev = sd.query_devices(idx)
            return AudioDevice(
                index=idx,
                name=dev['name'],
                channels_in=dev['max_input_channels'],
                channels_out=dev['max_output_channels'],
                sample_rate=int(dev['default_samplerate']),
                is_default=True
            )
        except:
            return None
    
    def get_default_output_device(self) -> Optional[AudioDevice]:
        if not self.is_available():
            return None
        
        try:
            idx = sd.default.device[1]
            if idx is None:
                return None
            dev = sd.query_devices(idx)
            return AudioDevice(
                index=idx,
                name=dev['name'],
                channels_in=dev['max_input_channels'],
                channels_out=dev['max_output_channels'],
                sample_rate=int(dev['default_samplerate']),
                is_default=True
            )
        except:
            return None
    
    def read_audio(self, device: AudioDevice, duration: float) -> np.ndarray:
        if not self.is_available():
            return np.array([])
        
        recording = sd.rec(
            int(duration * device.sample_rate),
            samplerate=device.sample_rate,
            channels=min(device.channels_in, 2),
            device=device.index,
            dtype=np.float32
        )
        sd.wait()
        return recording.flatten()
    
    def play_audio(self, device: AudioDevice, audio_data: np.ndarray, sample_rate: int):
        if not self.is_available():
            return
        
        sd.play(audio_data, samplerate=sample_rate, device=device.index)
        sd.wait()


class AudioHAL:
    """Main Audio Hardware Abstraction Layer"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "audio_config.json"
        self.config = self.load_config()
        self.platform = self.detect_platform()
        self.backends = self._initialize_backends()
        self.active_backend = None
        self.input_device = None
        self.output_device = None
        
        # Auto-configure on init
        self.auto_configure()
    
    def detect_platform(self) -> Dict[str, str]:
        """Detect current platform and hardware"""
        info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'python_version': platform.python_version()
        }
        
        # Check if we're on Jetson
        if os.path.exists('/etc/nv_tegra_release'):
            info['device_type'] = 'jetson'
            with open('/etc/nv_tegra_release', 'r') as f:
                info['jetson_info'] = f.read().strip()
        else:
            info['device_type'] = 'standard'
        
        return info
    
    def _initialize_backends(self) -> List[AudioBackend]:
        """Initialize all available audio backends"""
        backends = []
        
        # Try sounddevice first (usually more reliable)
        sd_backend = SoundDeviceBackend()
        if sd_backend.is_available():
            backends.append(sd_backend)
        
        # Then try pyaudio
        pa_backend = PyAudioBackend()
        if pa_backend.is_available():
            backends.append(pa_backend)
        
        return backends
    
    def load_config(self) -> Dict:
        """Load configuration from file if exists"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_config(self):
        """Save current configuration to file"""
        config = {
            'platform': self.platform,
            'active_backend': self.active_backend.__class__.__name__ if self.active_backend else None,
            'input_device': {
                'index': self.input_device.index,
                'name': self.input_device.name,
                'backend': self.input_device.backend
            } if self.input_device else None,
            'output_device': {
                'index': self.output_device.index,
                'name': self.output_device.name,
                'backend': self.output_device.backend
            } if self.output_device else None,
            'audio_settings': getattr(self, 'audio_settings', {})
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def auto_configure(self) -> bool:
        """Automatically configure audio based on platform"""
        # First, try to load saved config for this platform
        if self._load_platform_config():
            return True
        
        # Otherwise, auto-detect
        for backend in self.backends:
            self.active_backend = backend
            
            # Get default devices
            input_dev = backend.get_default_input_device()
            output_dev = backend.get_default_output_device()
            
            if input_dev and output_dev:
                self.input_device = input_dev
                self.output_device = output_dev
                
                # Platform-specific adjustments
                if self.platform['device_type'] == 'jetson':
                    # Jetson-specific settings
                    self.audio_settings = {
                        'gain': 50.0,  # From our testing
                        'threshold': 0.02,
                        'buffer_size': 1024
                    }
                else:
                    # Standard laptop settings
                    self.audio_settings = {
                        'gain': 1.0,  # Usually no gain needed
                        'threshold': 0.01,
                        'buffer_size': 512
                    }
                
                self.save_config()
                return True
        
        return False
    
    def _load_platform_config(self) -> bool:
        """Try to load config for current platform"""
        if not self.config:
            return False
        
        # Check if saved config matches current platform
        saved_platform = self.config.get('platform', {})
        if (saved_platform.get('hostname') == self.platform['hostname'] and
            saved_platform.get('system') == self.platform['system']):
            
            # Try to restore saved backend and devices
            backend_name = self.config.get('active_backend')
            for backend in self.backends:
                if backend.__class__.__name__ == backend_name:
                    self.active_backend = backend
                    
                    # Find saved devices
                    saved_input = self.config.get('input_device')
                    saved_output = self.config.get('output_device')
                    
                    if saved_input and saved_output:
                        devices = backend.list_devices()
                        
                        for dev in devices:
                            if dev.name == saved_input['name'] and dev.backend == saved_input.get('backend'):
                                self.input_device = dev
                            if dev.name == saved_output['name'] and dev.backend == saved_output.get('backend'):
                                self.output_device = dev
                        
                        if self.input_device and self.output_device:
                            self.audio_settings = self.config.get('audio_settings', {})
                            return True
        
        return False
    
    def list_all_devices(self) -> Dict[str, List[AudioDevice]]:
        """List devices from all available backends"""
        all_devices = {}
        for backend in self.backends:
            backend_name = backend.__class__.__name__
            all_devices[backend_name] = backend.list_devices()
        return all_devices
    
    def select_device(self, device_name: str, device_type: str = 'input') -> bool:
        """Select a specific device by name"""
        for backend in self.backends:
            devices = backend.list_devices()
            for dev in devices:
                if device_name.lower() in dev.name.lower():
                    if device_type == 'input' and dev.channels_in > 0:
                        self.input_device = dev
                        self.active_backend = backend
                        self.save_config()
                        return True
                    elif device_type == 'output' and dev.channels_out > 0:
                        self.output_device = dev
                        self.active_backend = backend
                        self.save_config()
                        return True
        return False
    
    def read_audio(self, duration: float = 1.0) -> Tuple[np.ndarray, int]:
        """Read audio from input device"""
        if not self.active_backend or not self.input_device:
            return np.array([]), 0
        
        audio_data = self.active_backend.read_audio(self.input_device, duration)
        
        # Apply gain if configured
        if hasattr(self, 'audio_settings') and 'gain' in self.audio_settings:
            audio_data = audio_data * self.audio_settings['gain']
        
        return audio_data, self.input_device.sample_rate
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: Optional[int] = None):
        """Play audio to output device"""
        if not self.active_backend or not self.output_device:
            return
        
        if sample_rate is None:
            sample_rate = self.output_device.sample_rate
        
        self.active_backend.play_audio(self.output_device, audio_data, sample_rate)
    
    def get_info(self) -> Dict:
        """Get current HAL configuration info"""
        return {
            'platform': self.platform,
            'available_backends': [b.__class__.__name__ for b in self.backends],
            'active_backend': self.active_backend.__class__.__name__ if self.active_backend else None,
            'input_device': str(self.input_device) if self.input_device else None,
            'output_device': str(self.output_device) if self.output_device else None,
            'audio_settings': getattr(self, 'audio_settings', {})
        }


def test_audio_hal():
    """Test the audio HAL"""
    print("=== Audio HAL Test ===\n")
    
    # Initialize HAL
    hal = AudioHAL()
    
    # Show platform info
    print("Platform Info:")
    for key, value in hal.platform.items():
        print(f"  {key}: {value}")
    
    print(f"\nDevice Type: {hal.platform['device_type']}")
    
    # Show HAL info
    print("\nHAL Configuration:")
    info = hal.get_info()
    for key, value in info.items():
        if key != 'platform':  # Already shown
            print(f"  {key}: {value}")
    
    # List all devices
    print("\nAvailable Audio Devices:")
    all_devices = hal.list_all_devices()
    for backend_name, devices in all_devices.items():
        print(f"\n{backend_name}:")
        for dev in devices:
            print(f"  {dev}")
    
    # Test audio if configured
    if hal.input_device and hal.output_device:
        print("\n=== Audio Test ===")
        print("Recording 2 seconds...")
        audio_data, sample_rate = hal.read_audio(2.0)
        
        if len(audio_data) > 0:
            max_val = np.max(np.abs(audio_data))
            print(f"Recorded {len(audio_data)} samples at {sample_rate}Hz")
            print(f"Max amplitude: {max_val:.3f}")
            
            if max_val > 0.001:
                print("Playing back...")
                hal.play_audio(audio_data, sample_rate)
            else:
                print("Audio too quiet to play back")
        else:
            print("No audio recorded")
    else:
        print("\n⚠️  No audio devices configured!")


if __name__ == "__main__":
    test_audio_hal()