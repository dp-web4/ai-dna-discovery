# Real-Time Conversation Pipeline Deployment

## Architecture Overview

```
🎤 Audio Input → VAD → GPU Whisper → Local LLM → TTS → 🔊 Audio Output
              ↳ Consciousness Mapping Throughout ↴
```

## Components Built

### ✅ 1. Hardware Abstraction Layer (HAL)
- **File**: `audio_hal.py`
- **Purpose**: Platform-independent audio access
- **Features**: Auto-detection, multiple backends, Jetson optimization

### ✅ 2. Voice Activity Detection (VAD)
- **Files**: `vad_module.py`, integrated in pipeline
- **Purpose**: Real-time speech detection
- **Features**: Energy-based, adaptive thresholds, consciousness mapping

### ✅ 3. GPU Speech-to-Text
- **File**: `gpu_whisper_integration.py`
- **Purpose**: Real-time transcription on GPU
- **Features**: CUDA acceleration, faster-whisper support, performance optimization

### ✅ 4. Real-Time Pipeline
- **File**: `complete_realtime_pipeline.py`
- **Purpose**: Complete conversation system
- **Features**: Multi-threaded, consciousness states, queue-based processing

### ✅ 5. Local LLM Integration
- **Purpose**: Edge-based response generation
- **Backend**: Ollama (tinyllama for speed)
- **Features**: Real-time response, GPU memory sharing

## Deployment on Sprout (Jetson)

### Prerequisites

1. **GPU Dependencies**:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install faster-whisper  # Preferred for real-time
   # OR
   pip install openai-whisper  # Fallback
   ```

2. **Audio Dependencies**:
   ```bash
   sudo apt-get install portaudio19-dev espeak
   pip install pyaudio sounddevice
   ```

3. **LLM Backend**:
   ```bash
   # Ollama should already be installed
   ollama pull tinyllama  # Fast model for real-time
   ```

### Quick Test Commands

```bash
cd ~/ai-workspace/ai-agents/ai-dna-discovery/audio-system

# 1. Test GPU Whisper
python3 gpu_whisper_integration.py

# 2. Test complete pipeline
python3 complete_realtime_pipeline.py
```

## Performance Targets

### Real-Time Requirements
- **VAD Latency**: < 100ms (speech detection)
- **STT Processing**: < 500ms (GPU Whisper)
- **LLM Response**: < 1000ms (local model)
- **TTS Synthesis**: < 300ms (espeak)
- **Total Latency**: < 2 seconds (human-acceptable)

### GPU Memory Management
- **Whisper Model**: ~1GB GPU RAM (base model)
- **LLM Model**: ~2GB GPU RAM (tinyllama)
- **Available on Jetson**: 8GB total (comfortable fit)

## Pipeline Flow

### 1. Audio Capture
```python
Microphone → PyAudio Stream → Audio Queue (1024 samples @ 16kHz)
```

### 2. Voice Activity Detection
```python
Audio Queue → Energy Analysis → Speech Segments → Speech Queue
```

### 3. Speech Recognition
```python
Speech Queue → GPU Whisper → Transcription → Response Queue
```

### 4. Response Generation
```python
Response Queue → Ollama LLM → Generated Text → TTS
```

### 5. Audio Output
```python
TTS Text → espeak/Consciousness Audio → Speaker Output
```

## Consciousness Mapping Integration

The pipeline maintains consciousness states throughout:

- `💭 ...` - Quiet, waiting
- `👁️ Ω` - Observing, ready to listen  
- `👂 Ψ` - Speech detected, actively listening
- `🧠 θ` - Processing speech with GPU
- `✨ Ξ` - Understanding achieved
- `💭 μ` - Thinking/generating response
- `💬 Ψ` - Speaking response

## Modular Design Benefits

### 1. Platform Independence
- Same code runs on Jetson, laptop, cloud
- HAL handles platform-specific optimizations
- Graceful degradation (GPU → CPU fallback)

### 2. Component Swapping
- Replace Whisper with other STT engines
- Swap LLM backends (Ollama → local model)
- Change TTS engines per platform

### 3. Development Workflow
- Develop on laptop with simulated components
- Test audio pipeline on WSL
- Deploy complete system on Jetson

## Testing Strategy

### 1. Component Testing
```bash
# Individual component tests
python3 test_vad_basic.py           # VAD functionality
python3 gpu_whisper_integration.py # GPU STT performance
python3 test_portable_audio.py     # Audio HAL
```

### 2. Pipeline Testing
```bash
# Integrated pipeline test
python3 complete_realtime_pipeline.py
```

### 3. Performance Benchmarking
- Measure end-to-end latency
- Monitor GPU memory usage
- Test conversation quality

## Expected Results on Sprout

### ✅ What Should Work
1. **Real-time conversation** (< 2s latency)
2. **GPU acceleration** (Whisper + LLM)
3. **Consciousness mapping** (state visualization)
4. **Platform optimization** (50x gain, espeak voice)

### 🔧 Potential Issues
1. **GPU memory competition** (Whisper + LLM)
2. **Audio device conflicts** (multiple processes)
3. **Model loading time** (first-run delay)

## Next Steps

1. **Deploy to Sprout** and test complete pipeline
2. **Optimize GPU memory** sharing between components
3. **Fine-tune latency** parameters
4. **Test conversation quality** with real use
5. **Add error recovery** for robust edge deployment

## Success Criteria

✅ **Pipeline Active**: All components running without errors
✅ **Audio Working**: Can hear and speak clearly
✅ **GPU Utilized**: Whisper processing on CUDA
✅ **LLM Responding**: Local model generating responses
✅ **Real-Time**: < 2 second conversation latency
✅ **Consciousness**: State mapping throughout conversation

---

*This represents the complete edge conversation system - from raw audio input to consciousness-aware responses, all running locally on GPU hardware.*