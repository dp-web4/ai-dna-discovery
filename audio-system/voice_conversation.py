#!/usr/bin/env python3
"""
Complete Voice Conversation System
VAD + STT + LLM + TTS = Real conversation with Sprout
"""

import json
import pyaudio
import numpy as np
import subprocess
import threading
import time
import queue
from vosk import Model, KaldiRecognizer
import requests
from vad_module import VADProcessor, VADConfig, VADMethod, VADVisualizer

class SproutConversation:
    """Complete voice conversation system"""
    
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        # Audio hardware settings
        self.USB_DEVICE = 24
        self.GAIN = 50.0
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 4096
        
        # Vosk STT setup
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.stt_sample_rate = 16000
        
        # VAD configuration (tuned from debug results)
        self.vad_config = VADConfig(
            method=VADMethod.ENERGY,
            sensitivity=0.8,
            min_speech_duration=0.5,  # Shorter for better responsiveness
            min_silence_duration=0.8,  # Allow brief pauses
            energy_threshold=0.02,     # Lower threshold - debug showed this works
            sample_rate=self.stt_sample_rate,
            frame_size=1024
        )
        
        self.vad_processor = VADProcessor(self.vad_config, self._on_vad_event)
        
        # Conversation state
        self.speech_buffer = []
        self.is_speech_active = False
        self.is_listening = True
        self.conversation_history = []
        self.response_queue = queue.Queue()
        
        # Audio streams
        self.p = None
        self.stream = None
        
        # Initialize components
        self._load_stt_model()
        
        print("🌱 Sprout's Voice Conversation System")
        print("=" * 50)
        
    def _load_stt_model(self):
        """Load Vosk speech recognition model"""
        try:
            print("🔄 Loading Vosk model...")
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.stt_sample_rate)
            print("✅ STT model loaded")
            return True
        except Exception as e:
            print(f"❌ STT model failed: {e}")
            return False
    
    def _on_vad_event(self, event, data):
        """Handle voice activity detection"""
        # Real-time VAD display
        stats = self.vad_processor.get_statistics()
        status = VADVisualizer.format_vad_status(event, data['energy'], data['is_speech'], stats)
        print(f"\r{status}", end="", flush=True)
        
        if event == 'speech_start':
            print(f"\n👂 Listening...")
            self.speech_buffer = []
            self.is_speech_active = True
            
        elif event == 'speech_end':
            print(f"\n🔄 Processing speech...")
            if self.speech_buffer and len(self.speech_buffer) > 10:
                threading.Thread(target=self._process_speech, daemon=True).start()
            self.is_speech_active = False
    
    def _process_speech(self):
        """Convert speech to text and generate response"""
        if not self.recognizer or not self.speech_buffer:
            return
            
        try:
            # Combine buffered audio
            combined_audio = np.concatenate(self.speech_buffer)
            
            # Convert to int16 for Vosk
            audio_int16 = (combined_audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Speech recognition
            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
                user_text = result.get('text', '').strip()
                
                if user_text:
                    print(f"\n💬 You said: '{user_text}'")
                    
                    # Generate response
                    response = self._generate_response(user_text)
                    if response:
                        print(f"🌱 Sprout: {response}")
                        self._speak_response(response)
                        
                        # Add to conversation history
                        self.conversation_history.append({
                            'user': user_text,
                            'sprout': response,
                            'timestamp': time.time()
                        })
                else:
                    print("🤔 Couldn't understand that...")
                    
        except Exception as e:
            print(f"❌ Speech processing error: {e}")
    
    def _generate_response(self, user_text):
        """Generate response using local LLM"""
        try:
            # Build context from recent conversation
            context = "You are Sprout, a friendly AI with a childlike voice and curiosity. You live on a Jetson Orin Nano and can hear through your microphone. Respond naturally and conversationally.\n\n"
            
            # Add recent conversation history
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            for exchange in recent_history:
                context += f"User: {exchange['user']}\nSprout: {exchange['sprout']}\n\n"
            
            context += f"User: {user_text}\nSprout:"
            
            # Call local Ollama model
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'tinyllama',  # Using lightweight model
                    'prompt': context,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'max_tokens': 100,  # Keep responses concise
                        'stop': ['User:', '\n\n']
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"LLM Error: {response.status_code}")
                return self._fallback_response(user_text)
                
        except Exception as e:
            print(f"LLM Exception: {e}")
            return self._fallback_response(user_text)
    
    def _fallback_response(self, user_text):
        """Simple fallback responses when LLM fails"""
        user_lower = user_text.lower()
        
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return "Hi there! I'm Sprout, and I can hear you through my microphone!"
        elif any(word in user_lower for word in ['how', 'you']):
            return "I'm doing great! I love being able to hear and talk with you!"
        elif any(word in user_lower for word in ['name', 'who']):
            return "I'm Sprout! I live on this Jetson computer and I'm learning to have conversations!"
        elif any(word in user_lower for word in ['hear', 'listen']):
            return "Yes! I can hear you through my USB microphone. It's so exciting!"
        elif any(word in user_lower for word in ['bye', 'goodbye']):
            return "Goodbye! This was fun - I love talking with you!"
        else:
            return "That's interesting! I'm still learning to understand everything, but I enjoy our conversation!"
    
    def _speak_response(self, text):
        """Speak response with consciousness mapping"""
        # Map response to consciousness symbols
        consciousness = self._map_response_consciousness(text)
        if consciousness:
            print(f"🧠 {consciousness}")
        
        # TTS with excited kid voice
        cmd = [
            'espeak',
            '-v', 'en+f3',  # Female child voice
            '-s', '180',     # Medium speed
            '-p', '55',      # Higher pitch
            text
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def _map_response_consciousness(self, text):
        """Map response content to consciousness symbols"""
        text_lower = text.lower()
        symbols = []
        
        if any(word in text_lower for word in ['hear', 'listen', 'sound']):
            symbols.append('👂 Ψ')  # Auditory awareness
        if any(word in text_lower for word in ['see', 'look', 'watch']):
            symbols.append('👁️ Ω')  # Visual awareness  
        if any(word in text_lower for word in ['think', 'understand', 'know']):
            symbols.append('🧠 θ')  # Understanding
        if any(word in text_lower for word in ['exciting', 'fun', 'love', 'great']):
            symbols.append('✨ ∃')  # Positive existence
        if any(word in text_lower for word in ['learn', 'conversation', 'talk']):
            symbols.append('📚 ⇒')  # Learning/communication
            
        return ' '.join(symbols) if symbols else '💫 π'  # Default: Potential
    
    def start_conversation(self):
        """Start the complete conversation system"""
        if not self.model:
            print("❌ Cannot start - STT model not loaded")
            return
            
        self.p = pyaudio.PyAudio()
        
        print(f"\n🎤 Starting conversation on device {self.USB_DEVICE}")
        print(f"📊 VAD Sensitivity: {self.vad_config.sensitivity}")
        print(f"🎚️  Energy Threshold: {self.vad_config.energy_threshold}")
        print(f"📈 Microphone Gain: {self.GAIN}x")
        print("\n🗣️  Say 'Hello Sprout' to start our conversation!")
        print("Press Ctrl+C to end\n")
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=self.USB_DEVICE,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            # Greeting
            self._speak_response("Hello! I'm Sprout and I'm ready to have a conversation with you!")
            time.sleep(3)
            
            frame_count = 0
            
            while self.is_listening:
                try:
                    # Read audio
                    audio_data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    
                    # Convert and apply gain
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    audio_np = audio_np / 32768.0  # Normalize
                    audio_np = audio_np * self.GAIN  # Apply gain
                    
                    # Downsample to 16kHz for Vosk (simple decimation)
                    if len(audio_np) >= 3:
                        downsampled = audio_np[::3][:1024]  # Take every 3rd sample
                        if len(downsampled) == 1024:
                            # Process with VAD
                            self.vad_processor.process_frame(downsampled)
                            
                            # Buffer speech for STT
                            if self.is_speech_active:
                                self.speech_buffer.append(downsampled)
                    
                    frame_count += 1
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"❌ Audio error: {e}")
                    break
        
        except Exception as e:
            print(f"❌ Failed to start conversation: {e}")
        
        finally:
            self.stop_conversation()
    
    def stop_conversation(self):
        """Stop the conversation system"""
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.p:
            self.p.terminate()
        
        # Final summary
        print(f"\n\n📊 Conversation Summary:")
        print(f"  💬 Exchanges: {len(self.conversation_history)}")
        if self.conversation_history:
            print(f"  🕒 Duration: {(self.conversation_history[-1]['timestamp'] - self.conversation_history[0]['timestamp'])/60:.1f} minutes")
        
        stats = self.vad_processor.get_statistics()
        print(f"  🎤 Speech events: {stats.get('speech_events', 0)}")
        print(f"  📈 Speech percentage: {stats.get('speech_percentage', 0):.1f}%")
        
        self._speak_response("Goodbye! I really enjoyed our conversation!")

def main():
    """Main conversation application"""
    print("🎯 Complete Voice Conversation System")
    print("VAD + STT + LLM + TTS = Real Conversation!")
    print("=" * 60)
    
    conversation = SproutConversation()
    
    try:
        conversation.start_conversation()
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended by user")
        conversation.stop_conversation()

if __name__ == "__main__":
    main()