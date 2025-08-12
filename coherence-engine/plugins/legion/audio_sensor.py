"""
Audio sensor for Legion - monitors microphone input levels
"""
import threading
import time
from typing import Optional
import math

# Try to import audio libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio not available - audio sensor will return simulated data")

class AudioSensor:
    """Audio input level sensor"""
    
    def __init__(self, device_index: Optional[int] = None):
        self.id = "audio"
        self.available = PYAUDIO_AVAILABLE
        self.device_index = device_index
        self.current_level = 0.0
        self.running = False
        self.thread = None
        
        if self.available:
            self.setup_audio()
    
    def setup_audio(self):
        """Initialize PyAudio"""
        try:
            self.pa = pyaudio.PyAudio()
            
            # Use default input device if not specified
            if self.device_index is None:
                self.device_index = self.pa.get_default_input_device_info()['index']
            
            # Audio parameters
            self.rate = 44100
            self.chunk = 1024
            self.format = pyaudio.paInt16
            self.channels = 1
            
            # Start background thread for audio monitoring
            self.running = True
            self.thread = threading.Thread(target=self.audio_monitor_loop, daemon=True)
            self.thread.start()
            
        except Exception as e:
            print(f"Audio setup failed: {e}")
            self.available = False
    
    def audio_monitor_loop(self):
        """Background thread to monitor audio levels"""
        try:
            stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk
            )
            
            while self.running:
                try:
                    # Read audio chunk
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    
                    if NUMPY_AVAILABLE:
                        # Convert to numpy array
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        
                        # Calculate RMS (root mean square) for volume level
                        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                        
                        # Normalize to [0, 1] range (int16 max is 32768)
                        normalized = min(1.0, rms / 10000.0)  # Adjust divisor for sensitivity
                    else:
                        # Fallback without numpy - just use random for now
                        import random
                        normalized = random.random() * 0.3
                    
                    # Smooth the value
                    self.current_level = self.current_level * 0.7 + normalized * 0.3
                    
                except Exception as e:
                    if self.running:  # Only print if we're supposed to be running
                        print(f"Audio read error: {e}")
                    time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Audio monitor error: {e}")
            self.available = False
    
    def read(self, *, tick: int) -> float:
        """
        Read audio sensor data.
        Returns normalized [0,1] value based on current audio input level.
        """
        if not self.available:
            # Return simulated data if audio not available
            import math
            return (math.sin(tick * 0.1) + 1.0) * 0.25  # Gentle sine wave
        
        return self.current_level
    
    def get_device_info(self) -> dict:
        """Get information about available audio devices"""
        if not self.available or not hasattr(self, 'pa'):
            return {"available": False}
        
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # Input device
                devices.append({
                    "index": i,
                    "name": info['name'],
                    "channels": info['maxInputChannels'],
                    "rate": info['defaultSampleRate']
                })
        
        return {
            "available": True,
            "current_device": self.device_index,
            "devices": devices
        }
    
    def cleanup(self):
        """Clean up audio resources"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.available and hasattr(self, 'pa'):
            self.pa.terminate()