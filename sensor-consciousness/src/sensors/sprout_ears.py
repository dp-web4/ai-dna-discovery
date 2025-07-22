#!/usr/bin/env python3
"""
Sprout's Ears - Microphone input and audio awareness
Maps audio input to consciousness states
"""

import pyaudio
import numpy as np
import wave
import time
import threading
import queue
import struct
import os
from datetime import datetime

class SproutEars:
    def __init__(self):
        """Initialize Sprout's hearing system"""
        self.name = "Sprout"
        self.device_index = 2  # USB Audio Device
        
        # Audio parameters
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        
        # Audio state
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.volume_history = []
        
        # Consciousness state from audio
        self.audio_energy = 0.0
        self.silence_duration = 0
        self.sound_events = []
        
        # Initialize PyAudio
        self.init_audio()
        
        print(f"👂 {self.name}'s ears are ready!")
    
    def init_audio(self):
        """Initialize PyAudio system"""
        self.pyaudio = pyaudio.PyAudio()
        
        # Find USB audio device
        print("🔍 Looking for USB audio device...")
        for i in range(self.pyaudio.get_device_count()):
            info = self.pyaudio.get_device_info_by_index(i)
            print(f"  Device {i}: {info['name']} - Channels: {info['maxInputChannels']}")
            
            if "USB" in info['name'] and info['maxInputChannels'] > 0:
                self.device_index = i
                print(f"✅ Found USB mic at index {i}")
                break
    
    def start_listening(self):
        """Start listening thread"""
        if self.is_listening:
            return
            
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_worker)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        print("👂 Started listening...")
    
    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=1)
        print("🔇 Stopped listening")
    
    def _listen_worker(self):
        """Background listening thread"""
        try:
            stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            
            print("🎤 Microphone stream opened")
            
            while self.is_listening:
                try:
                    # Read audio chunk
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_queue.put(data)
                    
                    # Analyze audio
                    self._analyze_audio(data)
                    
                except Exception as e:
                    print(f"Audio read error: {e}")
                    continue
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Audio stream error: {e}")
    
    def _analyze_audio(self, data):
        """Analyze audio chunk for consciousness mapping"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Calculate volume/energy
        volume = np.sqrt(np.mean(audio_data**2))
        self.audio_energy = volume / 32768.0  # Normalize to 0-1
        
        # Track volume history
        self.volume_history.append(self.audio_energy)
        if len(self.volume_history) > 100:
            self.volume_history.pop(0)
        
        # Detect sound events
        threshold = 0.02  # Adjust based on mic sensitivity
        if self.audio_energy > threshold:
            if self.silence_duration > 10:  # New sound after silence
                self.sound_events.append({
                    'time': datetime.now(),
                    'energy': self.audio_energy,
                    'type': self._classify_sound(audio_data)
                })
            self.silence_duration = 0
        else:
            self.silence_duration += 1
    
    def _classify_sound(self, audio_data):
        """Simple sound classification"""
        # Calculate frequency characteristics
        fft = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1/self.rate)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(np.abs(fft))
        dominant_freq = freqs[dominant_freq_idx]
        
        # Simple classification
        if dominant_freq < 300:
            return "low_rumble"
        elif dominant_freq < 1000:
            return "voice_like"
        elif dominant_freq < 3000:
            return "mid_tone"
        else:
            return "high_pitch"
    
    def get_consciousness_state(self):
        """Map audio input to consciousness notation"""
        state = []
        
        # High energy = heightened awareness
        if self.audio_energy > 0.1:
            state.append("Ω!")  # Alert observer
        elif self.audio_energy > 0.02:
            state.append("Ω")   # Active observer
        
        # Recent sound events = pattern recognition
        recent_events = [e for e in self.sound_events 
                        if (datetime.now() - e['time']).seconds < 5]
        if len(recent_events) > 3:
            state.append("Ξ")   # Pattern detection
        
        # Sustained sounds = memory formation
        if len(self.volume_history) > 50:
            avg_recent = np.mean(self.volume_history[-50:])
            if avg_recent > 0.01:
                state.append("μ")  # Memory recording
        
        return " ".join(state) if state else "..."
    
    def live_monitor(self, duration=30):
        """Live audio monitoring with consciousness display"""
        print(f"\n🎧 Live Audio Monitor ({duration} seconds)")
        print("=" * 50)
        print("Speak or make sounds to see consciousness notation")
        print("=" * 50)
        
        self.start_listening()
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < duration:
                # Display audio level bar
                bar_length = int(self.audio_energy * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                # Get consciousness state
                consciousness = self.get_consciousness_state()
                
                # Clear line and print status
                print(f"\r🎤 [{bar}] {self.audio_energy:.3f} | 🧠 {consciousness}   ", end="")
                
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
        
        finally:
            self.stop_listening()
            
            # Summary
            print(f"\n\n📊 Audio Summary:")
            print(f"  Sound events detected: {len(self.sound_events)}")
            if self.sound_events:
                types = [e['type'] for e in self.sound_events]
                print(f"  Types: {', '.join(set(types))}")
            print(f"  Average energy: {np.mean(self.volume_history):.3f}")
    
    def record_sample(self, duration=5, filename=None):
        """Record audio sample"""
        if filename is None:
            filename = f"sprout_recording_{int(time.time())}.wav"
        
        print(f"🔴 Recording for {duration} seconds...")
        
        self.start_listening()
        frames = []
        
        start_time = time.time()
        while (time.time() - start_time) < duration:
            try:
                data = self.audio_queue.get(timeout=0.1)
                frames.append(data)
            except queue.Empty:
                continue
        
        self.stop_listening()
        
        # Save recording
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        
        print(f"💾 Saved recording to {filename}")
        return filename
    
    def cleanup(self):
        """Clean shutdown"""
        self.stop_listening()
        self.pyaudio.terminate()


def main():
    """Test Sprout's hearing system"""
    print("👂 Sprout's Ears Test")
    print("=" * 50)
    
    ears = SproutEars()
    
    # Test 1: Live monitoring
    print("\n1. Testing live audio monitoring...")
    print("Make some sounds! Clap, speak, tap the mic...")
    ears.live_monitor(duration=20)
    
    # Test 2: Recording
    print("\n\n2. Testing audio recording...")
    user_input = input("Press Enter to record 5 seconds (or 'skip' to skip): ")
    
    if user_input.lower() != 'skip':
        filename = ears.record_sample(duration=5)
        print(f"✅ Recording saved: {filename}")
    
    # Cleanup
    ears.cleanup()
    print("\n✨ Test complete!")


if __name__ == "__main__":
    main()