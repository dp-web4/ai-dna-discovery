#!/usr/bin/env python3
"""
Audio Sensor with Bluetooth Integration
AIRHUG Bluetooth audio device with Web4-aligned confidence scoring
"""

import pyaudio
import numpy as np
import time
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from collections import deque
import threading
import queue

@dataclass
class AudioData:
    """Audio sensor data with confidence metrics"""
    timestamp: float
    audio_buffer: np.ndarray  # Raw audio samples
    amplitude: float  # RMS amplitude
    frequency_spectrum: np.ndarray  # FFT results
    dominant_frequency: float  # Most prominent frequency
    spectral_centroid: float  # "Brightness" of sound
    zero_crossing_rate: float  # Texture indicator
    bluetooth_rssi: Optional[int]  # Bluetooth signal strength
    latency: float  # Audio latency in ms
    confidence: float  # Overall audio confidence

class BluetoothManager:
    """Manage Bluetooth audio connection"""
    
    def __init__(self, device_name="AIRHUG 01"):
        self.device_name = device_name
        self.device_address = None
        self.is_connected = False
        self.rssi_history = deque(maxlen=10)
        
    def find_device(self) -> Optional[str]:
        """Find Bluetooth device by name"""
        try:
            # Scan for devices
            result = subprocess.run(
                ["bluetoothctl", "devices"],
                capture_output=True, text=True, check=True
            )
            
            # Parse output for our device
            for line in result.stdout.split('\n'):
                if self.device_name in line:
                    # Extract MAC address
                    parts = line.split()
                    if len(parts) >= 2:
                        self.device_address = parts[1]
                        print(f"Found {self.device_name} at {self.device_address}")
                        return self.device_address
            
            print(f"Device {self.device_name} not found in paired devices")
            return None
            
        except Exception as e:
            print(f"Error finding Bluetooth device: {e}")
            return None
    
    def connect(self) -> bool:
        """Connect to Bluetooth audio device"""
        if not self.device_address:
            self.device_address = self.find_device()
            if not self.device_address:
                return False
        
        try:
            # Connect to device
            result = subprocess.run(
                ["bluetoothctl", "connect", self.device_address],
                capture_output=True, text=True, timeout=10
            )
            
            if "Connection successful" in result.stdout:
                self.is_connected = True
                print(f"Connected to {self.device_name}")
                
                # Set as audio sink
                subprocess.run(
                    ["pactl", "set-default-sink", f"bluez_sink.{self.device_address.replace(':', '_')}"],
                    capture_output=True
                )
                
                # Set as audio source
                subprocess.run(
                    ["pactl", "set-default-source", f"bluez_source.{self.device_address.replace(':', '_')}"],
                    capture_output=True
                )
                
                return True
            else:
                print(f"Failed to connect: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"Error connecting to Bluetooth device: {e}")
            return False
    
    def get_rssi(self) -> Optional[int]:
        """Get Bluetooth signal strength (RSSI)"""
        if not self.is_connected or not self.device_address:
            return None
        
        try:
            # Get device info including RSSI
            result = subprocess.run(
                ["bluetoothctl", "info", self.device_address],
                capture_output=True, text=True
            )
            
            # Parse RSSI from output
            for line in result.stdout.split('\n'):
                if 'RSSI' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        rssi = int(parts[1].strip())
                        self.rssi_history.append(rssi)
                        return rssi
            
            return None
            
        except Exception as e:
            print(f"Error getting RSSI: {e}")
            return None
    
    def calculate_signal_confidence(self) -> float:
        """Calculate Bluetooth signal confidence"""
        if not self.rssi_history:
            return 0.5
        
        avg_rssi = np.mean(self.rssi_history)
        
        # RSSI typically ranges from -100 (worst) to -30 (best)
        if avg_rssi >= -30:
            return 1.0
        elif avg_rssi >= -50:
            return 0.9
        elif avg_rssi >= -70:
            return 0.7
        elif avg_rssi >= -85:
            return 0.5
        else:
            return 0.3

class AudioSensor:
    """Audio sensor with Bluetooth integration and analysis"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        
        # Bluetooth manager
        self.bluetooth = BluetoothManager()
        
        # Audio analysis buffers
        self.audio_queue = queue.Queue(maxsize=10)
        self.amplitude_history = deque(maxlen=30)
        self.frequency_history = deque(maxlen=30)
        self.confidence_history = deque(maxlen=10)
        
        # Latency measurement
        self.latency_measurements = deque(maxlen=10)
        
        # Background thread for audio capture
        self.capture_thread = None
    
    def connect(self) -> bool:
        """Connect to Bluetooth audio and start stream"""
        # Connect Bluetooth first
        if not self.bluetooth.connect():
            print("Warning: Bluetooth connection failed, using default audio")
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_audio)
            self.capture_thread.start()
            
            print(f"Audio stream opened at {self.sample_rate} Hz")
            return True
            
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            return False
    
    def _capture_audio(self):
        """Background thread to capture audio"""
        while self.is_running:
            try:
                # Read audio chunk
                start_time = time.time()
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # Store in queue with metadata
                if not self.audio_queue.full():
                    self.audio_queue.put({
                        'data': audio_data,
                        'timestamp': time.time(),
                        'latency': latency
                    })
                
                # Update latency measurements
                self.latency_measurements.append(latency)
                
            except Exception as e:
                print(f"Audio capture error: {e}")
                time.sleep(0.01)
    
    def analyze_audio(self, audio_data: np.ndarray) -> Dict:
        """Analyze audio features"""
        # Calculate RMS amplitude
        amplitude = np.sqrt(np.mean(audio_data**2))
        self.amplitude_history.append(amplitude)
        
        # Calculate FFT for frequency analysis
        fft = np.fft.rfft(audio_data)
        magnitude = np.abs(fft)
        frequencies = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
        
        # Find dominant frequency
        dominant_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
        dominant_frequency = frequencies[dominant_idx]
        self.frequency_history.append(dominant_frequency)
        
        # Calculate spectral centroid (brightness)
        spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
        
        # Calculate zero crossing rate (texture)
        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
        zero_crossing_rate = zero_crossings / len(audio_data)
        
        return {
            'amplitude': amplitude,
            'frequency_spectrum': magnitude,
            'dominant_frequency': dominant_frequency,
            'spectral_centroid': spectral_centroid,
            'zero_crossing_rate': zero_crossing_rate
        }
    
    def calculate_audio_confidence(self, features: Dict, bluetooth_conf: float) -> float:
        """Calculate overall audio confidence"""
        confidence_factors = []
        
        # Amplitude confidence (not too quiet, not clipping)
        amp = features['amplitude']
        if amp < 0.001:
            amp_conf = 0.3  # Too quiet
        elif amp > 0.9:
            amp_conf = 0.5  # Possible clipping
        else:
            amp_conf = 0.9  # Good level
        confidence_factors.append(amp_conf * 0.3)
        
        # Frequency stability (consistent dominant frequency)
        if len(self.frequency_history) >= 3:
            freq_std = np.std(list(self.frequency_history)[-3:])
            freq_conf = max(0, 1 - freq_std / 1000)  # Less variation is better
            confidence_factors.append(freq_conf * 0.2)
        else:
            confidence_factors.append(0.5 * 0.2)
        
        # Bluetooth connection confidence
        confidence_factors.append(bluetooth_conf * 0.3)
        
        # Latency confidence
        if self.latency_measurements:
            avg_latency = np.mean(self.latency_measurements)
            if avg_latency < 10:
                latency_conf = 1.0  # Excellent
            elif avg_latency < 30:
                latency_conf = 0.8  # Good
            elif avg_latency < 50:
                latency_conf = 0.6  # Acceptable
            else:
                latency_conf = 0.4  # Poor
            confidence_factors.append(latency_conf * 0.2)
        else:
            confidence_factors.append(0.7 * 0.2)
        
        # Calculate weighted confidence
        confidence = sum(confidence_factors)
        
        # Temporal smoothing
        self.confidence_history.append(confidence)
        return np.mean(self.confidence_history)
    
    def read(self) -> Optional[AudioData]:
        """Read and process audio data"""
        if not self.audio_queue.empty():
            # Get latest audio chunk
            audio_info = self.audio_queue.get()
            audio_buffer = audio_info['data']
            timestamp = audio_info['timestamp']
            latency = audio_info['latency']
            
            # Analyze audio features
            features = self.analyze_audio(audio_buffer)
            
            # Get Bluetooth RSSI
            rssi = self.bluetooth.get_rssi()
            bluetooth_conf = self.bluetooth.calculate_signal_confidence()
            
            # Calculate overall confidence
            confidence = self.calculate_audio_confidence(features, bluetooth_conf)
            
            # Create audio data object
            return AudioData(
                timestamp=timestamp,
                audio_buffer=audio_buffer,
                amplitude=features['amplitude'],
                frequency_spectrum=features['frequency_spectrum'],
                dominant_frequency=features['dominant_frequency'],
                spectral_centroid=features['spectral_centroid'],
                zero_crossing_rate=features['zero_crossing_rate'],
                bluetooth_rssi=rssi,
                latency=latency,
                confidence=confidence
            )
        
        return None
    
    def get_sensor_fusion_data(self) -> Dict:
        """Get data formatted for sensor fusion system"""
        data = self.read()
        if not data:
            return None
        
        return {
            'type': 'audio',
            'timestamp': data.timestamp,
            'data': {
                'amplitude': data.amplitude,
                'frequency': {
                    'dominant': data.dominant_frequency,
                    'centroid': data.spectral_centroid
                },
                'texture': data.zero_crossing_rate,
                'connection': {
                    'bluetooth_rssi': data.bluetooth_rssi,
                    'latency_ms': data.latency
                }
            },
            'confidence': data.confidence,
            'metadata': {
                'device': 'AIRHUG 01',
                'sample_rate': self.sample_rate,
                'connection': 'Bluetooth' if self.bluetooth.is_connected else 'Default'
            }
        }
    
    def close(self):
        """Close audio stream and Bluetooth connection"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        print("Audio sensor closed")

if __name__ == "__main__":
    # Test audio sensor
    audio = AudioSensor()
    
    if audio.connect():
        print("\nReading audio data...")
        print("-" * 60)
        
        try:
            for i in range(100):  # 10 seconds at ~10 Hz
                data = audio.read()
                if data:
                    print(f"Time: {data.timestamp:.2f}")
                    print(f"Amplitude: {data.amplitude:.4f}")
                    print(f"Dominant Freq: {data.dominant_frequency:.1f} Hz")
                    print(f"Spectral Centroid: {data.spectral_centroid:.1f} Hz")
                    if data.bluetooth_rssi:
                        print(f"Bluetooth RSSI: {data.bluetooth_rssi} dBm")
                    print(f"Latency: {data.latency:.1f} ms")
                    print(f"Confidence: {data.confidence:.2%}")
                    print("-" * 60)
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            audio.close()