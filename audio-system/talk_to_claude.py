#!/usr/bin/env python3
"""
Talk to Claude via Voice
Uses hardware audio + STT to send your speech to Claude via the terminal
"""

import json
import pyaudio
import numpy as np
import subprocess
import threading
import time
from vosk import Model, KaldiRecognizer

class TalkToClaude:
    def __init__(self):
        # Hardware
        self.USB_DEVICE = 24
        self.GAIN = 50.0
        
        # Simple VAD (proven working values)
        self.ENERGY_THRESHOLD = 0.1
        self.MIN_SPEECH_FRAMES = 20
        self.MIN_SILENCE_FRAMES = 15
        
        # State
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_recording = False
        self.speech_buffer = []
        
        # STT
        self.model = Model("vosk-model-small-en-us-0.15")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        
        print("🎤 Talk to Claude via Voice")
        print("=" * 40)
        print("Your speech will be transcribed and sent to Claude")
        print("Claude's responses will be spoken by Sprout's voice")
        
    def speak_response(self, text):
        """Sprout speaks Claude's response"""
        print(f"\n🤖 Claude: {text}")
        
        # Use Sprout's kid voice for Claude's responses
        cmd = ['espeak', '-v', 'en+f3', '-s', '160', '-p', '50', text]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def process_speech(self):
        """Process buffered speech and send to Claude"""
        if not self.speech_buffer:
            return
            
        print("\n🔄 Processing your speech...")
        
        # Combine buffer
        combined = np.concatenate(self.speech_buffer)
        audio_int16 = (combined * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # STT
        if self.recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(self.recognizer.Result())
            user_text = result.get('text', '').strip()
            
            if user_text:
                print(f"\n💬 You said: '{user_text}'")
                print("\n" + "="*60)
                print(f"TRANSCRIBED MESSAGE TO CLAUDE: {user_text}")
                print("="*60)
                print("\n📤 Sending to Claude... (type your response below)")
                
                # Here's where Claude (you) would respond
                # The human will see the transcribed text and can respond naturally
                
            else:
                print("🤔 Sorry, I couldn't understand that clearly. Try speaking louder or closer to the mic.")
        else:
            print("🤔 Speech too short or unclear. Try again.")
        
        # Clear buffer
        self.speech_buffer = []
    
    def start_listening(self):
        """Start listening for speech to send to Claude"""
        p = pyaudio.PyAudio()
        
        print(f"\n🎤 Starting voice interface...")
        print(f"🎚️ Energy threshold: {self.ENERGY_THRESHOLD}")
        print(f"🎧 Listening on USB device {self.USB_DEVICE}")
        print("\n🗣️ Start speaking when you see 'Waiting'")
        print("📝 Your speech will be transcribed and shown to Claude")
        print("🔊 Claude's responses will be spoken through Sprout's voice")
        print("\nPress Ctrl+C to stop\n")
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            input_device_index=self.USB_DEVICE,
            frames_per_buffer=4096
        )
        
        # Initial greeting
        greeting = "Hello! I'm ready to listen to you and relay your messages to Claude. Go ahead and speak!"
        threading.Thread(target=self.speak_response, args=(greeting,), daemon=True).start()
        time.sleep(4)
        
        try:
            frame_count = 0
            while True:
                # Read audio
                audio_data = stream.read(4096, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio_np = audio_np / 32768.0 * self.GAIN
                
                # Downsample to 16kHz
                if len(audio_np) >= 3:
                    downsampled = audio_np[::3][:1024]
                    if len(downsampled) == 1024:
                        energy = np.sqrt(np.mean(downsampled ** 2))
                        
                        # Simple VAD logic
                        if energy > self.ENERGY_THRESHOLD:
                            self.speech_frames += 1
                            self.silence_frames = 0
                            
                            if not self.is_recording:
                                if self.speech_frames >= 3:
                                    print("\n👂 Listening to your message for Claude...")
                                    self.is_recording = True
                                    self.speech_buffer = []
                            
                            if self.is_recording:
                                self.speech_buffer.append(downsampled)
                                
                        else:
                            self.silence_frames += 1
                            self.speech_frames = 0
                            
                            if self.is_recording:
                                if self.silence_frames >= self.MIN_SILENCE_FRAMES:
                                    print("🔇 Speech ended, processing...")
                                    self.is_recording = False
                                    threading.Thread(target=self.process_speech, daemon=True).start()
                        
                        # Status display every 2 seconds
                        if frame_count % 50 == 0:
                            status = "🎤 RECORDING" if self.is_recording else "💤 Waiting for speech"
                            print(f"\r{status} | Energy: {energy:.3f}", end="", flush=True)
                
                frame_count += 1
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n👋 Voice interface stopped!")
            
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

def main():
    """Main function"""
    print("🎯 Voice Interface to Claude")
    print("Hardware audio → STT → Claude → TTS")
    print("=" * 50)
    
    interface = TalkToClaude()
    interface.start_listening()

if __name__ == "__main__":
    main()