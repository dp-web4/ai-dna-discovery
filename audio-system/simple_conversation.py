#!/usr/bin/env python3
"""
Simple Conversation System
No complex VAD - just energy threshold + STT + LLM + TTS
"""

import json
import pyaudio
import numpy as np
import subprocess
import threading
import time
import requests
from vosk import Model, KaldiRecognizer

class SimpleConversation:
    def __init__(self):
        # Hardware
        self.USB_DEVICE = 24
        self.GAIN = 50.0
        
        # Simple VAD
        self.ENERGY_THRESHOLD = 0.1  # Simple threshold
        self.MIN_SPEECH_FRAMES = 20   # ~0.8 seconds
        self.MIN_SILENCE_FRAMES = 15  # ~0.6 seconds
        
        # State
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_recording = False
        self.speech_buffer = []
        
        # STT
        self.model = Model("vosk-model-small-en-us-0.15")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        
        print("🌱 Simple Conversation System")
        print(f"🎚️ Energy Threshold: {self.ENERGY_THRESHOLD}")
        print(f"📏 Min Speech: {self.MIN_SPEECH_FRAMES} frames")
        
    def speak(self, text):
        """Sprout speaks"""
        print(f"🌱 Sprout: {text}")
        cmd = ['espeak', '-v', 'en+f3', '-s', '180', '-p', '55', text]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except:
            pass
    
    def get_llm_response(self, user_text):
        """Get response from local LLM"""
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'tinyllama',
                    'prompt': f"You are Sprout, a friendly AI child. User said: '{user_text}'. Respond in 1-2 sentences as Sprout would.",
                    'stream': False
                },
                timeout=8
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except Exception as e:
            print(f"LLM error: {e}")
        
        # Fallback
        if 'hello' in user_text.lower():
            return "Hi! I'm Sprout and I can hear you!"
        elif 'how' in user_text.lower():
            return "I'm doing great! I love talking with you!"
        else:
            return "That's interesting! Tell me more!"
    
    def process_speech(self):
        """Process buffered speech"""
        if not self.speech_buffer:
            return
            
        print("🔄 Processing speech...")
        
        # Combine buffer
        combined = np.concatenate(self.speech_buffer)
        audio_int16 = (combined * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # STT
        if self.recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(self.recognizer.Result())
            user_text = result.get('text', '').strip()
            
            if user_text:
                print(f"💬 You: '{user_text}'")
                
                # Get LLM response
                response = self.get_llm_response(user_text)
                
                # Speak response
                threading.Thread(target=self.speak, args=(response,), daemon=True).start()
            else:
                print("🤔 Couldn't understand...")
        
        # Clear buffer
        self.speech_buffer = []
    
    def start_conversation(self):
        """Start simple conversation"""
        p = pyaudio.PyAudio()
        
        print("\n🎤 Starting simple conversation...")
        print("Speak clearly and I'll respond!")
        print("Press Ctrl+C to stop\n")
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            input_device_index=self.USB_DEVICE,
            frames_per_buffer=4096
        )
        
        # Greeting
        threading.Thread(target=self.speak, args=("Hello! I'm Sprout! Say something to me!",), daemon=True).start()
        time.sleep(3)
        
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
                                if self.speech_frames >= 3:  # Start recording after 3 speech frames
                                    print("\n👂 Started listening...")
                                    self.is_recording = True
                                    self.speech_buffer = []
                            
                            if self.is_recording:
                                self.speech_buffer.append(downsampled)
                                
                        else:
                            self.silence_frames += 1
                            self.speech_frames = 0
                            
                            if self.is_recording:
                                if self.silence_frames >= self.MIN_SILENCE_FRAMES:
                                    print("🔇 Processing speech...")
                                    self.is_recording = False
                                    threading.Thread(target=self.process_speech, daemon=True).start()
                        
                        # Status display
                        if frame_count % 50 == 0:
                            status = "🎤 RECORDING" if self.is_recording else "💤 Waiting"
                            print(f"\r{status} | Energy: {energy:.3f} | Threshold: {self.ENERGY_THRESHOLD}", end="", flush=True)
                
                frame_count += 1
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n👋 Conversation ended!")
            
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

def main():
    conversation = SimpleConversation()
    conversation.start_conversation()

if __name__ == "__main__":
    main()