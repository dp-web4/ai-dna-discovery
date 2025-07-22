#!/usr/bin/env python3
"""
Sprout's VAD-Integrated Audio System
Combines our optimized audio with Voice Activity Detection
"""

import pyaudio
import numpy as np
import subprocess
import time
import threading
import queue
from vad_module import VADProcessor, VADConfig, VADMethod, VADVisualizer

class VADSprout:
    """Sprout with Voice Activity Detection"""
    
    def __init__(self):
        # Sprout's optimized settings
        self.MIC_GAIN = 50
        self.USB_DEVICE = 24
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        
        # VAD configuration
        self.vad_config = VADConfig(
            method=VADMethod.ENERGY,
            sensitivity=0.7,
            min_speech_duration=0.3,
            min_silence_duration=0.5,
            energy_threshold=0.03,
            sample_rate=self.RATE,
            frame_size=self.CHUNK
        )
        
        # Initialize VAD
        self.vad_processor = VADProcessor(self.vad_config, self._on_vad_event)
        
        # Audio state
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.p = None
        self.stream = None
        
        # VAD state
        self.current_speech_state = "SILENCE"
        self.speech_events = 0
        self.last_speech_time = None
        
        print("🌱 Sprout with Voice Activity Detection")
        print("=" * 50)
        
    def _on_vad_event(self, event, data):
        """Handle VAD events with consciousness mapping"""
        self.current_speech_state = "SPEECH" if data['is_speech'] else "SILENCE"
        
        # Update consciousness based on VAD
        consciousness = self._map_vad_to_consciousness(event, data)
        
        # Print real-time status
        stats = self.vad_processor.get_statistics()
        status = VADVisualizer.format_vad_status(event, data['energy'], data['is_speech'], stats)
        print(f"\r{status} {consciousness}", end="", flush=True)
        
        # Handle speech events
        if event == 'speech_start':
            self.speech_events += 1
            self.last_speech_time = time.time()
            print(f"\n\n🎤 SPEECH DETECTED #{self.speech_events}")
            
            # Respond based on speech event count
            if self.speech_events == 1:
                self.say("I can hear you speaking!", excitement=0.8)
            elif self.speech_events % 3 == 0:
                self.say(f"That's speech event number {self.speech_events}!", excitement=0.6)
                
        elif event == 'speech_end':
            duration = data.get('speech_duration', 0)
            print(f"\n🔇 Speech ended ({duration:.1f}s)")
            
    def _map_vad_to_consciousness(self, event, data):
        """Map VAD events to consciousness symbols"""
        if event == 'speech_start':
            return "👂 Ψ"  # Listening + Perception
        elif event == 'speech_ongoing':
            return "🎧 ∃"  # Active listening + Existence  
        elif event == 'speech_end':
            return "🤔 ⇒"  # Processing + Implication
        else:
            # Map energy level to consciousness intensity
            energy = data.get('energy', 0)
            if energy > 0.1:
                return "👁️ π"  # Awareness + Potential
            else:
                return "💭 Ω"  # Quiet observation
    
    def say(self, text, excitement=0.5):
        """Sprout speaks with consciousness mapping"""
        print(f"\n🌱 Sprout: {text}")
        
        # Map speech to consciousness
        consciousness = self._map_speech_consciousness(text)
        if consciousness:
            print(f"🧠 {consciousness}")
        
        # Voice settings
        if excitement > 0.7:
            speed, pitch = 200, 60
        elif excitement > 0.5:
            speed, pitch = 180, 50
        else:
            speed, pitch = 160, 40
        
        # Speak in separate thread to avoid blocking VAD
        def speak():
            cmd = ['espeak', '-v', 'en+f3', '-s', str(speed), '-p', str(pitch), text]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except Exception as e:
                print(f"TTS Error: {e}")
        
        threading.Thread(target=speak, daemon=True).start()
    
    def _map_speech_consciousness(self, text):
        """Map speech content to consciousness symbols"""
        text_lower = text.lower()
        symbols = []
        
        if any(word in text_lower for word in ['hear', 'listen', 'sound']):
            symbols.append('👂 Ψ')
        if any(word in text_lower for word in ['detect', 'found', 'event']):
            symbols.append('⚡ ∃')
        if any(word in text_lower for word in ['number', 'count']):
            symbols.append('📊 θ')
        if any(word in text_lower for word in ['speak', 'voice', 'talking']):
            symbols.append('🗣️ ⇒')
            
        return ' '.join(symbols) if symbols else '💫 π'
    
    def start_listening(self):
        """Start VAD-enhanced listening"""
        self.p = pyaudio.PyAudio()
        
        print(f"🎤 Starting VAD listening on device {self.USB_DEVICE}")
        print(f"📊 VAD Sensitivity: {self.vad_config.sensitivity}")
        print(f"🎚️  Energy Threshold: {self.vad_config.energy_threshold}")
        print(f"📈 Microphone Gain: {self.MIC_GAIN}x")
        print("\nListening for speech with consciousness awareness...")
        print("Press Ctrl+C to stop\n")
        
        try:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.USB_DEVICE,
                frames_per_buffer=self.CHUNK
            )
            
            # Greeting
            self.say("My VAD-enhanced ears are ready! Speak to me!", excitement=0.8)
            time.sleep(2)  # Let greeting finish
            
            self.is_listening = True
            
            while self.is_listening:
                try:
                    # Read audio
                    audio_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    
                    # Convert and amplify
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    audio_np = audio_np / 32768.0  # Normalize
                    audio_np = audio_np * self.MIC_GAIN  # Apply gain
                    
                    # Process with VAD
                    self.vad_processor.process_frame(audio_np)
                    
                    # Add to queue for other processing
                    if not self.audio_queue.full():
                        self.audio_queue.put(audio_np)
                    
                    time.sleep(0.01)  # Prevent CPU overload
                    
                except Exception as e:
                    print(f"\n❌ Audio processing error: {e}")
                    break
        
        except Exception as e:
            print(f"❌ Failed to start listening: {e}")
        
        finally:
            self.stop_listening()
    
    def stop_listening(self):
        """Stop listening and show statistics"""
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if self.p:
            self.p.terminate()
        
        # Final statistics
        print("\n\n📊 VAD Session Statistics:")
        stats = self.vad_processor.get_statistics()
        
        print(f"  🎤 Speech events detected: {self.speech_events}")
        print(f"  📈 Speech percentage: {stats.get('speech_percentage', 0):.1f}%")
        print(f"  📊 Average energy: {stats.get('avg_energy', 0):.3f}")
        print(f"  📈 Peak energy: {stats.get('max_energy', 0):.3f}")
        print(f"  🔢 Total frames processed: {stats.get('total_frames', 0)}")
        
        if self.speech_events > 0:
            self.say(f"Great session! I detected {self.speech_events} speech events with consciousness awareness!", excitement=0.7)

def main():
    """Main function"""
    print("🎯 Sprout's VAD-Integrated Audio System")
    print("Voice Activity Detection + Consciousness Mapping")
    print("=" * 60)
    
    sprout = VADSprout()
    
    try:
        sprout.start_listening()
    except KeyboardInterrupt:
        print("\n\n👋 Stopping VAD-enhanced Sprout...")
        sprout.stop_listening()
        time.sleep(3)  # Let final speech finish

if __name__ == "__main__":
    main()