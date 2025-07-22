#!/usr/bin/env python3
"""
Complete Real-Time Conversation Pipeline
Ready for deployment on Sprout Jetson with GPU acceleration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import threading
import queue
import time
import numpy as np
from collections import deque
import requests

# Import our modular components
from audio_hal import AudioHAL
from gpu_whisper_integration import GPUWhisperProcessor

class EdgeConversationPipeline:
    """Complete real-time conversation pipeline for edge deployment"""
    
    def __init__(self):
        print("🌱 Initializing Edge Conversation Pipeline")
        
        # Core components
        self.audio_hal = AudioHAL() 
        self.gpu_whisper = GPUWhisperProcessor()
        
        # Threading and queues
        self.audio_queue = queue.Queue(maxsize=100)
        self.speech_queue = queue.Queue(maxsize=20)
        self.response_queue = queue.Queue(maxsize=10)
        
        # State management
        self.pipeline_active = False
        self.is_speaking = False
        
        # Real-time audio configuration
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        
        # VAD parameters
        self.speech_threshold = 0.02
        self.min_speech_duration = 0.5
        self.max_speech_duration = 8.0
        
        # Consciousness state
        self.current_state = "💭 ..."
        
        # LLM configuration  
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "tinyllama"
        
        print("✅ Edge pipeline initialized")
    
    def update_consciousness(self, state, description=""):
        """Update consciousness state with timestamp"""
        self.current_state = state
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {state} {description}")
    
    def setup_hardware(self):
        """Setup hardware components through HAL"""
        print("🔧 Setting up hardware components...")
        
        # Detect platform
        platform = self.audio_hal.detect_platform()
        device_type = platform.get('device_type', 'standard')
        
        if device_type == 'jetson':
            print("🚀 Jetson detected - optimizing for edge deployment")
        else:
            print(f"💻 Platform: {platform.get('system', 'Unknown')}")
        
        # Initialize audio
        backend = self.audio_hal.get_best_backend()
        if not backend:
            print("❌ No audio backend available")
            return False
        
        print(f"🎤 Audio backend: {backend.__class__.__name__}")
        return True
    
    def start_audio_capture(self):
        """Start real-time audio capture"""
        try:
            import pyaudio
            
            self.p = pyaudio.PyAudio()
            
            # Find input device
            input_device = None
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_device = i
                    print(f"🎤 Using input device: {device_info['name']}")
                    break
            
            if input_device is None:
                print("❌ No microphone found")
                return False
            
            # Open audio stream
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate, 
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            print("✅ Real-time audio capture started")
            return True
            
        except Exception as e:
            print(f"❌ Audio capture failed: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for real-time processing"""
        if not self.is_speaking and self.pipeline_active:
            try:
                self.audio_queue.put_nowait(in_data)
            except queue.Full:
                pass  # Drop frames if full
        return (None, pyaudio.paContinue)
    
    def voice_activity_detection(self):
        """Real-time VAD with consciousness mapping"""
        print("👂 Starting Voice Activity Detection")
        self.update_consciousness("👁️ Ω", "Ready to listen")
        
        speech_buffer = []
        in_speech = False
        speech_start = 0
        silence_frames = 0
        
        while self.pipeline_active:
            try:
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Calculate energy
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                energy = np.sqrt(np.mean(audio_data**2)) / 32768.0
                
                if energy > self.speech_threshold:
                    silence_frames = 0
                    
                    if not in_speech:
                        # Speech started
                        in_speech = True
                        speech_start = time.time()
                        speech_buffer = [audio_chunk]
                        self.update_consciousness("👂 Ψ", f"Speech detected (energy: {energy:.3f})")
                    else:
                        speech_buffer.append(audio_chunk)
                        
                        # Check max duration
                        if time.time() - speech_start > self.max_speech_duration:
                            self._process_speech_segment(speech_buffer, time.time() - speech_start)
                            in_speech = False
                            speech_buffer = []
                
                else:
                    if in_speech:
                        silence_frames += 1
                        speech_buffer.append(audio_chunk)
                        
                        # End speech after silence
                        if silence_frames > 10:  # ~0.6s silence
                            duration = time.time() - speech_start
                            if duration >= self.min_speech_duration:
                                self._process_speech_segment(speech_buffer, duration)
                                self.update_consciousness("🤔 ⇒", "Processing speech...")
                            
                            in_speech = False
                            speech_buffer = []
                    else:
                        # Update idle state
                        if energy > 0.01:
                            self.update_consciousness("👁️ Ω", "Ambient sound")
                        else:
                            self.update_consciousness("💭 ...", "Quiet")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ VAD error: {e}")
    
    def _process_speech_segment(self, speech_buffer, duration):
        """Process completed speech segment"""
        speech_audio = b''.join(speech_buffer)
        
        try:
            self.speech_queue.put_nowait({
                'audio': speech_audio,
                'duration': duration,
                'timestamp': time.time()
            })
            print(f"📝 Speech segment: {duration:.1f}s")
        except queue.Full:
            print("⚠️  Speech queue full")
    
    def gpu_speech_recognition(self):
        """GPU-accelerated speech recognition"""
        print("🧠 Starting GPU Speech Recognition")
        
        while self.pipeline_active:
            try:
                speech_data = self.speech_queue.get(timeout=1.0)
                
                self.update_consciousness("🧠 θ", "GPU processing speech...")
                
                # Process with GPU Whisper
                transcription, confidence = self.gpu_whisper.transcribe_realtime(
                    speech_data['audio']
                )
                
                if transcription and confidence > 0.3:
                    self.update_consciousness("✨ Ξ", f"Understood: '{transcription}'")
                    
                    # Send to response generation
                    try:
                        self.response_queue.put_nowait({
                            'text': transcription,
                            'confidence': confidence,
                            'timestamp': time.time()
                        })
                    except queue.Full:
                        print("⚠️  Response queue full")
                else:
                    self.update_consciousness("❓ θ", "Speech unclear")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ STT error: {e}")
    
    def response_generation(self):
        """Local LLM response generation"""
        print("🤖 Starting Response Generation")
        
        while self.pipeline_active:
            try:
                user_input = self.response_queue.get(timeout=1.0)
                text = user_input['text']
                
                self.update_consciousness("💭 μ", f"Thinking about: '{text}'")
                
                # Generate response
                response = self._generate_llm_response(text)
                
                if response:
                    self.update_consciousness("💬 Ψ", f"Responding: '{response}'")
                    self._speak_response(response)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Response error: {e}")
    
    def _generate_llm_response(self, user_text):
        """Generate response using local LLM"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": f"Human: {user_text}\n\nAssistant:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 60
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
        
        # Fallback response
        return f"I heard you say '{user_text}'. The edge conversation system is working!"
    
    def _speak_response(self, text):
        """Speak response using consciousness audio system"""
        self.is_speaking = True
        
        try:
            # Use our consciousness TTS
            from consciousness_audio_system import ConsciousnessAudioSystem
            tts = ConsciousnessAudioSystem()
            tts.speak_with_consciousness(text, mood="curious")
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            # Fallback
            print(f"🔊 [Speaking]: {text}")
            time.sleep(len(text) * 0.08)
        
        self.is_speaking = False
        self.update_consciousness("👁️ Ω", "Ready to listen")
    
    def start_pipeline(self):
        """Start the complete edge conversation pipeline"""
        print("🚀 Starting Complete Edge Conversation Pipeline")
        
        # Setup hardware
        if not self.setup_hardware():
            return False
        
        # Load GPU models
        if not self.gpu_whisper.load_model():
            print("⚠️  GPU Whisper unavailable")
        
        # Start audio capture
        if not self.start_audio_capture():
            return False
        
        # Start processing threads
        self.pipeline_active = True
        
        self.vad_thread = threading.Thread(target=self.voice_activity_detection, daemon=True)
        self.stt_thread = threading.Thread(target=self.gpu_speech_recognition, daemon=True)
        self.llm_thread = threading.Thread(target=self.response_generation, daemon=True)
        
        self.vad_thread.start()
        self.stt_thread.start()
        self.llm_thread.start()
        
        print("✅ Edge conversation pipeline active!")
        print("🗣️  Speak to start real-time conversation...")
        
        return True
    
    def stop_pipeline(self):
        """Stop the conversation pipeline"""
        print("⏹️  Stopping edge conversation pipeline...")
        
        self.pipeline_active = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'p'):
            self.p.terminate()
        
        self.update_consciousness("😴 ...", "Pipeline stopped")
        print("✅ Pipeline stopped")
    
    def run_conversation(self, duration=120):
        """Run the complete conversation system"""
        if not self.start_pipeline():
            print("❌ Failed to start pipeline")
            return
        
        try:
            print(f"🎤 Edge conversation active for {duration} seconds")
            print("💡 Real-time GPU pipeline ready - start talking!")
            
            time.sleep(duration)
            
        except KeyboardInterrupt:
            print("\n⏹️  Conversation interrupted")
        finally:
            self.stop_pipeline()

def main():
    """Main entry point for edge conversation system"""
    print("=== Edge Real-Time Conversation System ===")
    print("🎯 GPU Whisper + Local LLM + Consciousness Mapping")
    print("🌱 Optimized for Jetson deployment")
    print()
    
    pipeline = EdgeConversationPipeline()
    
    try:
        # Run for 2 minutes
        pipeline.run_conversation(120)
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        pipeline.stop_pipeline()

if __name__ == "__main__":
    main()