#!/usr/bin/env python3
"""
Real-time GPU-based conversation pipeline
Audio → VAD → GPU STT → LLM → TTS → Audio
"""

import pyaudio
import numpy as np
import threading
import queue
import time
import requests
import json
from collections import deque

class RealTimeConversationPipeline:
    def __init__(self):
        # Audio configuration
        self.CHUNK_SIZE = 1024
        self.SAMPLE_RATE = 16000  # Standard for speech processing
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Pipeline components
        self.audio_queue = queue.Queue(maxsize=50)  # Ring buffer for audio chunks
        self.speech_queue = queue.Queue(maxsize=10)  # Detected speech segments
        self.response_queue = queue.Queue(maxsize=5)  # LLM responses
        
        # State management
        self.is_listening = False
        self.is_speaking = False
        self.pipeline_active = False
        
        # VAD state
        self.vad_buffer = deque(maxlen=20)  # 20 chunks = ~1.3 seconds at 1024/16000
        self.speech_threshold = 0.02
        self.silence_threshold = 0.01
        self.min_speech_chunks = 8  # ~0.5 seconds
        self.min_silence_chunks = 6  # ~0.4 seconds
        
        self.speech_chunks = 0
        self.silence_chunks = 0
        self.in_speech = False
        self.speech_buffer = []
        
        # LLM configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "tinyllama"  # Fast model for real-time
        
        print("🎯 Real-time GPU conversation pipeline initialized")
    
    def init_audio(self):
        """Initialize PyAudio for real-time streaming"""
        try:
            self.p = pyaudio.PyAudio()
            
            # Find input device
            input_device = None
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"🎤 Found input device: {device_info['name']}")
                    if input_device is None:  # Use first available
                        input_device = i
            
            if input_device is None:
                print("❌ No input device found")
                return False
            
            # Open input stream
            self.stream_in = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            
            print(f"✅ Audio input initialized - Device {input_device}")
            return True
            
        except Exception as e:
            print(f"❌ Audio initialization failed: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Real-time audio callback - runs in separate thread"""
        if not self.is_speaking:  # Don't record while speaking
            try:
                self.audio_queue.put_nowait(in_data)
            except queue.Full:
                pass  # Drop frames if queue full
        return (None, pyaudio.paContinue)
    
    def energy_vad(self, audio_chunk):
        """Simple energy-based Voice Activity Detection"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_data**2)) / 32768.0  # Normalize
        
        return energy
    
    def vad_processor(self):
        """Voice Activity Detection processor thread"""
        print("🎧 VAD processor started")
        
        while self.pipeline_active:
            try:
                # Get audio chunk
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Calculate energy
                energy = self.energy_vad(audio_chunk)
                self.vad_buffer.append((audio_chunk, energy))
                
                # VAD state machine
                if energy > self.speech_threshold:
                    self.speech_chunks += 1
                    self.silence_chunks = 0
                    
                    if not self.in_speech and self.speech_chunks >= self.min_speech_chunks:
                        # Speech started
                        self.in_speech = True
                        self.speech_buffer = list(self.vad_buffer)  # Include context
                        print("🗣️  Speech started")
                        
                elif energy < self.silence_threshold:
                    self.silence_chunks += 1
                    self.speech_chunks = max(0, self.speech_chunks - 1)
                    
                    if self.in_speech and self.silence_chunks >= self.min_silence_chunks:
                        # Speech ended
                        self.in_speech = False
                        
                        # Extract audio data
                        speech_audio = b''.join([chunk for chunk, _ in self.speech_buffer])
                        
                        try:
                            self.speech_queue.put_nowait(speech_audio)
                            print(f"📝 Speech segment captured ({len(speech_audio)} bytes)")
                        except queue.Full:
                            print("⚠️  Speech queue full, dropping segment")
                        
                        self.speech_buffer = []
                
                # Continue collecting if in speech
                if self.in_speech:
                    self.speech_buffer.append((audio_chunk, energy))
                    
                    # Prevent infinite speech segments
                    if len(self.speech_buffer) > 80:  # ~5 seconds max
                        print("⏰ Max speech length reached")
                        self.in_speech = False
                        speech_audio = b''.join([chunk for chunk, _ in self.speech_buffer])
                        try:
                            self.speech_queue.put_nowait(speech_audio)
                        except queue.Full:
                            pass
                        self.speech_buffer = []
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ VAD error: {e}")
    
    def gpu_stt_processor(self):
        """GPU-based Speech-to-Text processor (placeholder for Whisper)"""
        print("🧠 GPU STT processor started")
        
        while self.pipeline_active:
            try:
                speech_audio = self.speech_queue.get(timeout=1.0)
                
                # TODO: Replace with GPU Whisper
                # For now, simulate processing
                print("🔄 Processing speech with GPU STT...")
                time.sleep(0.1)  # Simulate GPU processing time
                
                # Placeholder - return simulated transcription
                transcription = "Hello Claude, I'm testing the real-time pipeline"
                confidence = 0.85
                
                print(f"📝 Transcribed: '{transcription}' (confidence: {confidence})")
                
                # Send to LLM processor
                try:
                    self.response_queue.put_nowait({
                        'text': transcription,
                        'confidence': confidence,
                        'timestamp': time.time()
                    })
                except queue.Full:
                    print("⚠️  Response queue full")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ STT error: {e}")
    
    def llm_processor(self):
        """Local LLM processor using Ollama"""
        print("🤖 LLM processor started")
        
        while self.pipeline_active:
            try:
                speech_data = self.response_queue.get(timeout=1.0)
                user_text = speech_data['text']
                
                print(f"💭 Generating response to: '{user_text}'")
                
                # Query local Ollama
                response_text = self.query_ollama(user_text)
                
                if response_text:
                    print(f"🔊 Response: '{response_text}'")
                    # TODO: Send to TTS processor
                    self.speak_response(response_text)
                else:
                    print("❌ No response from LLM")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ LLM error: {e}")
    
    def query_ollama(self, user_input):
        """Query local Ollama model"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": f"User: {user_input}\n\nAssistant:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 100  # Keep responses short for real-time
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"Ollama error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Ollama connection error: {e}")
            return None
    
    def speak_response(self, text):
        """Speak response (placeholder for real TTS)"""
        self.is_speaking = True
        print(f"🔊 Speaking: '{text}'")
        
        # TODO: Replace with actual TTS
        time.sleep(len(text) * 0.05)  # Simulate speaking time
        
        self.is_speaking = False
    
    def start_pipeline(self):
        """Start the real-time conversation pipeline"""
        print("🚀 Starting real-time conversation pipeline...")
        
        if not self.init_audio():
            return False
        
        self.pipeline_active = True
        self.is_listening = True
        
        # Start processing threads
        self.vad_thread = threading.Thread(target=self.vad_processor, daemon=True)
        self.stt_thread = threading.Thread(target=self.gpu_stt_processor, daemon=True)
        self.llm_thread = threading.Thread(target=self.llm_processor, daemon=True)
        
        self.vad_thread.start()
        self.stt_thread.start()
        self.llm_thread.start()
        
        self.stream_in.start_stream()
        
        print("✅ Pipeline active - speak to start conversation!")
        return True
    
    def stop_pipeline(self):
        """Stop the conversation pipeline"""
        print("⏹️  Stopping pipeline...")
        
        self.pipeline_active = False
        self.is_listening = False
        
        if hasattr(self, 'stream_in'):
            self.stream_in.stop_stream()
            self.stream_in.close()
        
        if hasattr(self, 'p'):
            self.p.terminate()
        
        print("✅ Pipeline stopped")
    
    def run(self, duration_seconds=30):
        """Run the pipeline for specified duration"""
        if not self.start_pipeline():
            return
        
        try:
            print(f"🎤 Listening for {duration_seconds} seconds...")
            print("💡 Say something to test the real-time conversation!")
            
            time.sleep(duration_seconds)
            
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        finally:
            self.stop_pipeline()

def main():
    print("=== Real-Time GPU Conversation Pipeline ===\n")
    
    pipeline = RealTimeConversationPipeline()
    
    try:
        pipeline.run(30)  # Run for 30 seconds
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        pipeline.stop_pipeline()

if __name__ == "__main__":
    main()