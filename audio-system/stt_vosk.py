#!/usr/bin/env python3
"""
Speech-to-Text using Vosk
Real-time speech recognition for conversation
"""

import json
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer
import os
import time
from vad_module import VADProcessor, VADConfig, VADMethod

class VoskSTT:
    """Real-time Speech-to-Text using Vosk"""
    
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.sample_rate = 16000
        
        # Audio settings for Jetson
        self.USB_DEVICE = 24
        self.GAIN = 50.0
        
        # Initialize model
        self._load_model()
        
        # VAD for better speech detection
        self.vad_config = VADConfig(
            method=VADMethod.ENERGY,
            sensitivity=0.7,
            min_speech_duration=0.5,  # Longer for better STT chunks
            min_silence_duration=0.8,
            energy_threshold=0.03,
            sample_rate=self.sample_rate,
            frame_size=4096  # Larger for STT
        )
        
        self.vad_processor = VADProcessor(self.vad_config, self._on_vad_event)
        
        # Speech buffering
        self.speech_buffer = []
        self.is_speech_active = False
        self.last_transcript = ""
        
        print("🎤 Vosk Speech-to-Text Initialized")
        print(f"📂 Model: {model_path}")
        print(f"📊 Sample Rate: {self.sample_rate}Hz")
        
    def _load_model(self):
        """Load Vosk model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            print("Download with:")
            print("curl -L -o vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
            print("unzip vosk-model-small-en-us-0.15.zip")
            return False
            
        try:
            print("🔄 Loading Vosk model...")
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            print("✅ Vosk model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def _on_vad_event(self, event, data):
        """Handle VAD events for speech buffering"""
        if event == 'speech_start':
            print("\n🎤 Speech detected - starting buffer...")
            self.speech_buffer = []
            self.is_speech_active = True
            
        elif event == 'speech_end':
            print("🔇 Speech ended - processing buffer...")
            if self.speech_buffer and len(self.speech_buffer) > 10:  # Minimum length
                self._process_speech_buffer()
            self.is_speech_active = False
            self.speech_buffer = []
    
    def _process_speech_buffer(self):
        """Process accumulated speech buffer"""
        if not self.recognizer or not self.speech_buffer:
            return
            
        print("🔄 Processing speech...")
        
        # Combine buffered audio
        combined_audio = np.concatenate(self.speech_buffer)
        
        # Convert to bytes for Vosk (16-bit PCM)
        audio_int16 = (combined_audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Process with Vosk
        try:
            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '').strip()
                
                if text and text != self.last_transcript:
                    self.last_transcript = text
                    print(f"\n💬 TRANSCRIBED: '{text}'")
                    return text
            else:
                # Partial result
                partial = json.loads(self.recognizer.PartialResult())
                partial_text = partial.get('partial', '').strip()
                if partial_text:
                    print(f"🔄 Partial: '{partial_text}'")
                    
        except Exception as e:
            print(f"❌ STT Error: {e}")
        
        return None
    
    def start_listening(self):
        """Start real-time speech recognition"""
        if not self.model:
            print("❌ No model loaded")
            return
            
        p = pyaudio.PyAudio()
        
        print(f"\n🎤 Starting STT on device {self.USB_DEVICE}")
        print("Speak clearly and I'll transcribe your speech!")
        print("Press Ctrl+C to stop\n")
        
        try:
            # Use lower sample rate for Vosk
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,  # Jetson rate
                input=True,
                input_device_index=self.USB_DEVICE,
                frames_per_buffer=4096
            )
            
            frame_count = 0
            
            while True:
                try:
                    # Read audio
                    audio_data = stream.read(4096, exception_on_overflow=False)
                    
                    # Convert and apply gain
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    audio_np = audio_np / 32768.0  # Normalize
                    audio_np = audio_np * self.GAIN  # Apply gain
                    
                    # Downsample from 44100 to 16000 for Vosk
                    # Simple decimation (take every ~3rd sample)
                    if len(audio_np) >= 3:
                        downsampled = audio_np[::3][:self.vad_config.frame_size]
                        if len(downsampled) == self.vad_config.frame_size:
                            # Process with VAD
                            self.vad_processor.process_frame(downsampled)
                            
                            # Buffer speech for STT
                            if self.is_speech_active:
                                self.speech_buffer.append(downsampled)
                    
                    frame_count += 1
                    if frame_count % 100 == 0:  # Status every ~4 seconds
                        stats = self.vad_processor.get_statistics()
                        print(f"📊 Processed {frame_count} frames, {stats.get('speech_events', 0)} speech events")
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"❌ Audio processing error: {e}")
                    break
        
        except Exception as e:
            print(f"❌ Failed to start listening: {e}")
        
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()
            
            # Final stats
            stats = self.vad_processor.get_statistics()
            print(f"\n📊 Final STT Statistics:")
            print(f"  🎤 Speech events: {stats.get('speech_events', 0)}")
            print(f"  📈 Speech percentage: {stats.get('speech_percentage', 0):.1f}%")
            print(f"  🔄 Frames processed: {frame_count}")

def main():
    """Test STT functionality"""
    print("🎯 Vosk Speech-to-Text Test")
    print("=" * 40)
    
    stt = VoskSTT()
    
    if stt.model:
        try:
            stt.start_listening()
        except KeyboardInterrupt:
            print("\n\n👋 STT test stopped")
    else:
        print("❌ Cannot start - model not loaded")

if __name__ == "__main__":
    main()