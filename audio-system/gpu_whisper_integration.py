#!/usr/bin/env python3
"""
GPU Whisper integration for real-time STT
Designed for Jetson deployment with CUDA acceleration
"""

import numpy as np
import time
import io
import wave

class GPUWhisperProcessor:
    """GPU-accelerated Whisper for real-time speech recognition"""
    
    def __init__(self, model_size="base", device="cuda"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.sample_rate = 16000
        
        print(f"🧠 Initializing GPU Whisper ({model_size}) on {device}")
    
    def load_model(self):
        """Load Whisper model with GPU acceleration"""
        try:
            # Try faster-whisper first (optimized for real-time)
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(
                    self.model_size, 
                    device=self.device,
                    compute_type="float16"  # Use half precision for speed
                )
                self.model_type = "faster_whisper"
                print("✅ Using faster-whisper for optimal performance")
                
            except ImportError:
                # Fallback to OpenAI Whisper
                import whisper
                self.model = whisper.load_model(self.model_size)
                if self.device == "cuda":
                    self.model = self.model.cuda()
                self.model_type = "openai_whisper"
                print("✅ Using OpenAI Whisper (consider installing faster-whisper)")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def audio_to_numpy(self, audio_bytes):
        """Convert audio bytes to numpy array for Whisper"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 and normalize
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        return audio_float
    
    def transcribe_realtime(self, audio_bytes):
        """Transcribe audio bytes in real-time"""
        if self.model is None:
            if not self.load_model():
                return None, 0.0
        
        try:
            # Convert audio to proper format
            audio_array = self.audio_to_numpy(audio_bytes)
            
            # Ensure minimum length (Whisper needs at least ~0.1s)
            if len(audio_array) < self.sample_rate * 0.1:
                return None, 0.0
            
            start_time = time.time()
            
            if self.model_type == "faster_whisper":
                # faster-whisper API
                segments, info = self.model.transcribe(
                    audio_array, 
                    beam_size=1,  # Faster beam search
                    language="en",
                    condition_on_previous_text=False  # Better for real-time
                )
                
                # Collect segments
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
                
                transcription = " ".join(text_parts).strip()
                confidence = info.language_probability if hasattr(info, 'language_probability') else 0.8
                
            else:
                # OpenAI Whisper API
                result = self.model.transcribe(
                    audio_array,
                    language="english",
                    fp16=(self.device == "cuda")  # Use half precision on GPU
                )
                
                transcription = result["text"].strip()
                # OpenAI Whisper doesn't provide confidence, use segments if available
                confidence = 0.8  # Default confidence
                if "segments" in result and result["segments"]:
                    # Average confidence from segments
                    confidences = [seg.get("confidence", 0.8) for seg in result["segments"]]
                    confidence = np.mean(confidences)
            
            processing_time = time.time() - start_time
            
            if transcription:
                print(f"🎯 GPU STT: '{transcription}' ({processing_time:.2f}s, conf: {confidence:.2f})")
                return transcription, confidence
            else:
                return None, 0.0
                
        except Exception as e:
            print(f"❌ GPU STT error: {e}")
            return None, 0.0
    
    def benchmark_performance(self, test_duration=3.0):
        """Benchmark GPU performance with synthetic audio"""
        print(f"⚡ Benchmarking GPU Whisper performance...")
        
        # Generate test audio (sine wave with some noise)
        samples = int(self.sample_rate * test_duration)
        t = np.linspace(0, test_duration, samples)
        
        # Create synthetic speech-like signal
        signal = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # Base tone
            0.2 * np.sin(2 * np.pi * 880 * t) +  # Harmonic
            0.1 * np.random.randn(samples)       # Noise
        )
        
        # Convert to int16 bytes
        audio_int16 = (signal * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Benchmark transcription
        start_time = time.time()
        transcription, confidence = self.transcribe_realtime(audio_bytes)
        processing_time = time.time() - start_time
        
        real_time_factor = test_duration / processing_time if processing_time > 0 else 0
        
        print(f"📊 Performance Results:")
        print(f"   Audio duration: {test_duration:.1f}s")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Real-time factor: {real_time_factor:.1f}x")
        print(f"   Result: '{transcription}' (conf: {confidence:.2f})")
        
        if real_time_factor >= 1.0:
            print("✅ GPU performance suitable for real-time conversation!")
        else:
            print("⚠️  GPU performance may be too slow for real-time")
        
        return real_time_factor

def test_gpu_whisper():
    """Test GPU Whisper integration"""
    print("=== GPU Whisper Integration Test ===\n")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 CUDA GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = "cpu"
            print("💻 Using CPU (CUDA not available)")
    except ImportError:
        device = "cpu"
        print("💻 PyTorch not available, using CPU")
    
    # Initialize processor
    processor = GPUWhisperProcessor(model_size="base", device=device)
    
    # Load model
    if not processor.load_model():
        print("❌ Failed to load Whisper model")
        return
    
    # Benchmark performance
    processor.benchmark_performance(3.0)
    
    print(f"\n🎯 GPU Whisper ready for real-time pipeline integration!")

if __name__ == "__main__":
    test_gpu_whisper()