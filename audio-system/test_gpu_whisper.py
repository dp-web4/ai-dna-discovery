#!/usr/bin/env python3
"""
Test GPU-accelerated Whisper performance
"""

import os
import time
import whisper
import torch

def test_gpu_whisper():
    """Test Whisper with GPU acceleration"""
    
    print("=== GPU-Accelerated Whisper Test ===\n")
    
    # Check environment
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    print(f"🔧 CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB\n")
    
    # Load model with GPU
    print("🧠 Loading Whisper base model on GPU...")
    start_time = time.time()
    model = whisper.load_model("base", device="cuda")
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    print(f"✅ Model device: {model.device}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with sample audio (silence)
    print("\n🎤 Testing with sample audio...")
    import numpy as np
    
    # Generate 3 seconds of sample audio (simulated speech)
    sample_rate = 16000
    duration = 3
    samples = np.random.normal(0, 0.1, sample_rate * duration).astype(np.float32)
    
    # Add some speech-like patterns
    for i in range(0, len(samples), 1000):
        samples[i:i+500] *= np.sin(np.linspace(0, 10*np.pi, 500)) * 2
    
    print(f"📊 Audio sample: {len(samples)} samples, {duration}s duration")
    
    # Transcribe with timing
    print("\n🚀 Running GPU transcription...")
    start_time = time.time()
    result = model.transcribe(samples)
    transcription_time = time.time() - start_time
    
    print(f"⏱️ Transcription completed in {transcription_time:.2f} seconds")
    print(f"📝 Result: '{result['text'].strip()}'")
    print(f"🎯 Language detected: {result.get('language', 'unknown')}")
    
    # Performance metrics
    audio_duration = len(samples) / sample_rate
    real_time_factor = audio_duration / transcription_time
    
    print(f"\n📊 Performance Metrics:")
    print(f"   • Audio duration: {audio_duration:.2f}s")
    print(f"   • Processing time: {transcription_time:.2f}s")
    print(f"   • Real-time factor: {real_time_factor:.2f}x")
    print(f"   • {'🚀 Faster than real-time!' if real_time_factor > 1 else '⏳ Slower than real-time'}")
    
    print(f"\n🎉 GPU-accelerated Whisper test completed successfully!")
    return True

if __name__ == "__main__":
    # Set CUDA library path
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.6/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    try:
        test_gpu_whisper()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()