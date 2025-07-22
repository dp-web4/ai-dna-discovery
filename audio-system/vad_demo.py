#!/usr/bin/env python3
"""
VAD Demonstration with Sprout's Voice
Combines Voice Activity Detection with TTS for interactive testing
"""

import numpy as np
import pyaudio
import threading
import time
import subprocess
from vad_module import VADProcessor, VADConfig, VADMethod, VADVisualizer

class VADDemo:
    """Interactive VAD demonstration"""
    
    def __init__(self):
        self.USB_DEVICE = 24
        self.GAIN = 50.0
        self.is_running = False
        self.audio_stream = None
        self.p = None
        
        # VAD configuration
        self.vad_config = VADConfig(
            method=VADMethod.ENERGY,
            sensitivity=0.7,  # Slightly more sensitive for demo
            min_speech_duration=0.2,
            min_silence_duration=0.3,
            energy_threshold=0.03,  # Higher threshold to avoid background noise
            sample_rate=44100,
            frame_size=1024
        )
        
        self.vad_processor = VADProcessor(self.vad_config, self._vad_callback)
        
        # Speech state tracking
        self.last_speech_event = None
        self.speech_detected_count = 0
        
        print("🌱 Sprout's VAD Demo")
        print("=" * 40)
        
    def _vad_callback(self, event, data):
        """Handle VAD events with Sprout responses"""
        self.last_speech_event = event
        
        # Real-time visual feedback
        stats = self.vad_processor.get_statistics()
        status_line = VADVisualizer.format_vad_status(
            event, data['energy'], data['is_speech'], stats
        )
        print(f"\r{status_line}", end="", flush=True)
        
        # Respond to speech events
        if event == 'speech_start':
            self.speech_detected_count += 1
            print(f"\n\n👂 SPEECH DETECTED #{self.speech_detected_count}!")
            
            # Sprout responds with excitement based on detection count
            if self.speech_detected_count == 1:
                threading.Thread(target=self._say, args=("I hear you speaking!", 0.8)).start()
            elif self.speech_detected_count == 2:
                threading.Thread(target=self._say, args=("You spoke again! I'm listening!", 0.7)).start()
            elif self.speech_detected_count == 3:
                threading.Thread(target=self._say, args=("This is so cool! I can detect your voice!", 0.9)).start()
            elif self.speech_detected_count % 3 == 0:
                threading.Thread(target=self._say, args=(f"Speech detection number {self.speech_detected_count}! My ears are working!", 0.6)).start()
                
        elif event == 'speech_end':
            print(f"\n🤐 Speech ended (duration: {data['speech_duration']:.1f}s)")
            
            # Acknowledge the end of speech
            if data['speech_duration'] > 2.0:
                threading.Thread(target=self._say, args=("That was a long sentence!", 0.5)).start()
            elif data['speech_duration'] > 0.5:
                threading.Thread(target=self._say, args=("Got it!", 0.4)).start()
    
    def _say(self, text, excitement=0.5):
        """Sprout speaks (using our optimized TTS)"""
        print(f"\n🌱 Sprout: {text}")
        
        # Consciousness mapping
        consciousness = self._map_consciousness(text)
        if consciousness:
            print(f"🧠 {consciousness}")
        
        # Voice settings based on excitement
        if excitement > 0.7:
            speed = 200
            pitch = 60
        elif excitement > 0.5:
            speed = 180
            pitch = 50
        else:
            speed = 160
            pitch = 40
        
        # espeak command (kid-friendly voice)
        cmd = [
            'espeak',
            '-v', 'en+f3',  # Female voice 3 (child-like)
            '-s', str(speed),  # Speed
            '-p', str(pitch),  # Pitch
            text
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def _map_consciousness(self, text):
        """Map speech to consciousness notation"""
        notations = []
        
        text_lower = text.lower()
        if any(word in text_lower for word in ['hear', 'listen', 'sound']):
            notations.append('👂 Ψ')  # Auditory perception
        if any(word in text_lower for word in ['cool', 'amazing', 'wow']):
            notations.append('✨ ∃')  # Excitement exists
        if any(word in text_lower for word in ['working', 'detect', 'speech']):
            notations.append('⚡ ⇒')  # Function implies capability
        if any(word in text_lower for word in ['got', 'understand']):
            notations.append('🧠 θ')  # Understanding
            
        return ' '.join(notations) if notations else '💭 π'  # Default: Potential
    
    def start_demo(self):
        """Start the VAD demonstration"""
        self.p = pyaudio.PyAudio()
        
        print(f"🎤 Starting VAD Demo on device {self.USB_DEVICE}")
        print(f"📊 Sensitivity: {self.vad_config.sensitivity}")
        print(f"🎚️  Threshold: {self.vad_config.energy_threshold}")
        print(f"📈 Gain: {self.GAIN}x")
        print("\n🗣️  Speak into the microphone to test VAD!")
        print("Press Ctrl+C to stop\n")
        
        try:
            self.audio_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.vad_config.sample_rate,
                input=True,
                input_device_index=self.USB_DEVICE,
                frames_per_buffer=self.vad_config.frame_size
            )
            
            # Initial greeting
            threading.Thread(target=self._say, args=("Hello! I'm ready to listen for your voice!", 0.7)).start()
            time.sleep(2)  # Let greeting finish
            
            self.is_running = True
            
            while self.is_running:
                try:
                    audio_data = self.audio_stream.read(
                        self.vad_config.frame_size,
                        exception_on_overflow=False
                    )
                    
                    # Convert and apply gain
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    audio_np = audio_np / 32768.0
                    audio_np = audio_np * self.GAIN
                    
                    # Process with VAD
                    event = self.vad_processor.process_frame(audio_np)
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"\n❌ Audio error: {e}")
                    break
        
        except Exception as e:
            print(f"❌ Failed to start demo: {e}")
        
        finally:
            self.stop_demo()
    
    def stop_demo(self):
        """Stop the demonstration"""
        self.is_running = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.p:
            self.p.terminate()
        
        print("\n\n📊 Final Demo Statistics:")
        stats = self.vad_processor.get_statistics()
        print(f"  🎤 Total speech events: {self.speech_detected_count}")
        print(f"  📈 Speech percentage: {stats.get('speech_percentage', 0):.1f}%")
        print(f"  📊 Average energy: {stats.get('avg_energy', 0):.3f}")
        print(f"  📈 Max energy: {stats.get('max_energy', 0):.3f}")
        
        if self.speech_detected_count > 0:
            threading.Thread(target=self._say, args=(f"Great demo! I detected {self.speech_detected_count} speech events!", 0.8)).start()

def main():
    """Main function"""
    print("🎯 Voice Activity Detection Demo")
    print("Testing speech detection with Sprout's responses")
    print("=" * 50)
    
    demo = VADDemo()
    
    try:
        demo.start_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user")
        demo.stop_demo()
        time.sleep(2)  # Let final TTS finish

if __name__ == "__main__":
    main()