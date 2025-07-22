#!/usr/bin/env python3
"""
Whisper-Enhanced Conversation System
Combines our proven VAD/audio with better Whisper STT
"""

import whisper
import pyaudio
import numpy as np
import subprocess
import threading
import time
import tempfile
import wave
import requests

class WhisperConversation:
    def __init__(self):
        # Hardware (proven working settings)
        self.USB_DEVICE = 24
        self.GAIN = 50.0
        
        # VAD (adjusted for your microphone)
        self.ENERGY_THRESHOLD = 0.08  # Lowered to better detect speech
        self.MIN_SPEECH_FRAMES = 10   # Reduced for quicker response
        self.MIN_SILENCE_FRAMES = 8    # Reduced for faster processing
        
        # State
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_recording = False
        self.speech_buffer = []
        
        # Load Whisper model with GPU acceleration
        print("🧠 Loading Whisper model with GPU acceleration...")
        self.whisper_model = whisper.load_model("base", device="cuda")  # GPU-accelerated
        print(f"✅ GPU-accelerated Whisper model loaded on {self.whisper_model.device}")
        print("🚀 Using NVIDIA Orin GPU for speech recognition!")
        
        print("🌱 Whisper-Enhanced Conversation System")
        print(f"🎚️ Energy Threshold: {self.ENERGY_THRESHOLD}")
        
    def speak_response(self, text):
        """Sprout speaks response"""
        print(f"\n🌱 Sprout: {text}")
        
        # Use proven kid voice settings
        cmd = ['espeak', '-v', 'en+f3', '-s', '170', '-p', '52', text]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def get_response(self, user_text):
        """Get LLM response (local or Claude)"""
        try:
            # Try local LLM first
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'tinyllama',
                    'prompt': f"You are Sprout, a friendly AI child on Jetson. User said: '{user_text}'. Respond naturally in 1-2 sentences.",
                    'stream': False
                },
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except:
            pass
        
        # Fallback responses
        user_lower = user_text.lower()
        if 'hello' in user_lower or 'hi' in user_lower:
            return "Hello! I can hear you much better with Whisper now!"
        elif 'how' in user_lower and 'you' in user_lower:
            return "I'm doing great! My speech recognition got a big upgrade!"
        elif 'whisper' in user_lower:
            return "Yes! I'm using OpenAI Whisper now - much better than before!"
        elif 'test' in user_lower:
            return "This is working wonderfully! The audio quality is so much better!"
        else:
            return f"I heard you say '{user_text}' - that's much clearer than before!"
    
    def process_speech_with_whisper(self):
        """Process buffered speech with Whisper"""
        if not self.speech_buffer:
            return
            
        print("\n🔄 Processing speech with Whisper...")
        
        try:
            # Combine buffer
            combined = np.concatenate(self.speech_buffer)
            
            # Convert to int16 for WAV
            audio_int16 = (combined * 32767).clip(-32767, 32767).astype(np.int16)
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(44100)  # Use our actual sample rate
                    wav_file.writeframes(audio_int16.tobytes())
                
                # Process with Whisper
                print("🧠 Whisper processing...")
                result = self.whisper_model.transcribe(temp_file.name)
                user_text = result["text"].strip()
                
                # Clean up temp file
                import os
                os.unlink(temp_file.name)
                
                if user_text:
                    print(f"\n💬 Whisper heard: '{user_text}'")
                    
                    # Generate response
                    response = self.get_response(user_text)
                    
                    # Speak response
                    threading.Thread(target=self.speak_response, args=(response,), daemon=True).start()
                else:
                    print("🤔 Whisper couldn't understand that")
                    
        except Exception as e:
            print(f"❌ Whisper processing error: {e}")
        
        # Clear buffer
        self.speech_buffer = []
    
    def start_conversation(self):
        """Start Whisper-enhanced conversation"""
        p = pyaudio.PyAudio()
        
        print(f"\n🎤 Starting Whisper conversation...")
        print(f"🎧 Listening on USB device {self.USB_DEVICE}")
        print("🧠 Using OpenAI Whisper for better speech recognition")
        print("\n🗣️ Speak clearly and I'll understand you better!")
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
        threading.Thread(target=self.speak_response, args=("Hello! I'm Sprout with Whisper-enhanced speech recognition! Try talking to me!",), daemon=True).start()
        time.sleep(4)
        
        try:
            frame_count = 0
            while True:
                # Read audio
                audio_data = stream.read(4096, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio_np = audio_np / 32768.0 * self.GAIN
                
                # Simple VAD (same as our working version)
                energy = np.sqrt(np.mean(audio_np ** 2))
                
                if energy > self.ENERGY_THRESHOLD:
                    self.speech_frames += 1
                    self.silence_frames = 0
                    
                    if not self.is_recording:
                        if self.speech_frames >= 3:
                            print("\n👂 Listening with Whisper...")
                            self.is_recording = True
                            self.speech_buffer = []
                    
                    if self.is_recording:
                        self.speech_buffer.append(audio_np)
                        
                else:
                    self.silence_frames += 1
                    
                    if self.is_recording:
                        # Keep appending during silence to capture trailing audio
                        self.speech_buffer.append(audio_np)
                        
                        if self.silence_frames >= self.MIN_SILENCE_FRAMES:
                            print("🔇 Processing with Whisper...")
                            self.is_recording = False
                            self.silence_frames = 0
                            # Process in background thread to avoid blocking
                            threading.Thread(target=self.process_speech_with_whisper, daemon=True).start()
                    else:
                        self.speech_frames = 0
                
                # Status display
                if frame_count % 50 == 0:
                    status = "🎤 RECORDING" if self.is_recording else "💤 Waiting"
                    silence_info = f" | Silence: {self.silence_frames}/{self.MIN_SILENCE_FRAMES}" if self.is_recording else ""
                    print(f"\r{status} | Energy: {energy:.3f}{silence_info} | Whisper Ready", end="", flush=True)
                
                frame_count += 1
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n👋 Whisper conversation ended!")
            
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

def main():
    """Main function"""
    print("🎯 Whisper-Enhanced Voice Conversation")
    print("Better STT with proven audio pipeline")
    print("=" * 50)
    
    conversation = WhisperConversation()
    conversation.start_conversation()

if __name__ == "__main__":
    main()