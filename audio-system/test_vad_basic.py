#!/usr/bin/env python3
"""
Basic VAD Testing Script
Tests Voice Activity Detection with real microphone input
"""

import numpy as np
import pyaudio
import threading
import time
import sys
from vad_module import VADProcessor, VADConfig, VADMethod, VADVisualizer

class VADTester:
    """Real-time VAD testing with microphone"""
    
    def __init__(self, device_index=None):
        self.device_index = device_index
        self.is_running = False
        self.audio_stream = None
        self.p = None
        
        # VAD configuration (using Jetson-compatible settings)
        self.vad_config = VADConfig(
            method=VADMethod.ENERGY,
            sensitivity=0.6,
            min_speech_duration=0.3,
            min_silence_duration=0.5,
            energy_threshold=0.02,
            sample_rate=44100,  # Use working sample rate from Sprout
            frame_size=1024     # Use working frame size from Sprout
        )
        
        # Initialize VAD processor
        self.vad_processor = VADProcessor(self.vad_config, self._vad_callback)
        
        # Apply Jetson-specific gain (from our previous work)
        self.gain = 50.0  # Optimized gain for Jetson USB mic
        
    def _vad_callback(self, event, data):
        """Handle VAD events"""
        stats = self.vad_processor.get_statistics()
        status_line = VADVisualizer.format_vad_status(
            event, data['energy'], data['is_speech'], stats
        )
        
        # Clear line and print status
        print(f"\r{status_line}", end="", flush=True)
        
        # Print event on new line for significant changes
        if event in ['speech_start', 'speech_end']:
            print(f"\n🔊 {event.upper()} detected!")
            
    def find_microphone(self):
        """Find USB microphone device"""
        self.p = pyaudio.PyAudio()
        
        print("Available audio devices:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  {i}: {info['name']} (inputs: {info['maxInputChannels']})")
        
        # Try to auto-detect USB device or use provided index
        if self.device_index is None:
            # Look for USB audio device (from our previous work)
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0 and 'USB' in info['name'].upper():
                    self.device_index = i
                    print(f"\n✅ Auto-detected USB microphone: {info['name']}")
                    break
        
        if self.device_index is None:
            # Use default input device
            default_input = self.p.get_default_input_device_info()
            self.device_index = default_input['index']
            print(f"\n📱 Using default input: {default_input['name']}")
        
        return self.device_index
    
    def start_testing(self):
        """Start real-time VAD testing"""
        if not self.find_microphone():
            print("❌ No microphone found!")
            return
        
        try:
            # Open audio stream (using int16 format like sprout_optimized_audio.py)
            self.audio_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.vad_config.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.vad_config.frame_size
            )
            
            print(f"\n🎤 VAD Testing Started")
            print(f"Device: {self.device_index}")
            print(f"Method: {self.vad_config.method.value}")
            print(f"Sensitivity: {self.vad_config.sensitivity}")
            print(f"Threshold: {self.vad_config.energy_threshold}")
            print(f"Gain: {self.gain}x")
            print("\nSpeak into the microphone to test VAD...")
            print("Press Ctrl+C to stop\n")
            
            self.is_running = True
            
            while self.is_running:
                try:
                    # Read audio data
                    audio_data = self.audio_stream.read(
                        self.vad_config.frame_size,
                        exception_on_overflow=False
                    )
                    
                    # Convert to numpy array and apply gain (int16 to float32)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    audio_np = audio_np / 32768.0  # Normalize to [-1, 1]
                    audio_np = audio_np * self.gain
                    
                    # Process with VAD
                    event = self.vad_processor.process_frame(audio_np)
                    
                    # Small delay to prevent CPU overload
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"\n❌ Audio processing error: {e}")
                    break
        
        except Exception as e:
            print(f"❌ Failed to start audio stream: {e}")
        
        finally:
            self.stop_testing()
    
    def stop_testing(self):
        """Stop VAD testing"""
        self.is_running = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.p:
            self.p.terminate()
        
        print("\n\n📊 Final VAD Statistics:")
        stats = self.vad_processor.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

def main():
    """Main function"""
    print("🎯 Voice Activity Detection (VAD) Basic Test")
    print("=" * 50)
    
    # Check for device index argument
    device_index = None
    if len(sys.argv) > 1:
        try:
            device_index = int(sys.argv[1])
            print(f"Using specified device index: {device_index}")
        except ValueError:
            print("Invalid device index, using auto-detection")
    
    # Create and run tester
    tester = VADTester(device_index)
    
    try:
        tester.start_testing()
    except KeyboardInterrupt:
        print("\n\n👋 VAD test stopped by user")
        tester.stop_testing()

if __name__ == "__main__":
    main()