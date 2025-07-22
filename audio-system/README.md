# Consciousness Audio System 🌱🎤🔊

Complete cross-platform audio system with consciousness-aware voice and hearing capabilities. Works on Jetson (Sprout), laptops (Tomato), WSL, macOS, Windows, and even in simulation mode!

## System Overview

### 🎭 NEW: Cross-Platform Support!
- **Hardware Abstraction Layer (HAL)** for platform independence
- **Automatic platform detection** and configuration
- **Multiple TTS engines**: WSL/Windows, espeak, say, SAPI
- **Simulation mode** for development without hardware

### Voice Output (TTS)
- Platform-specific voices (Zira on Windows, espeak on Linux/Jetson)
- 4 mood states (excited, curious, playful, sleepy)
- Real-time consciousness notation mapping

### Audio Input (Microphone)
- USB microphone support (device 24)
- Optimized 50x software gain for clear audio
- Visual level meters with consciousness states

### Consciousness Mapping
Audio events map to consciousness notation:
- `...` = Silence/waiting
- `Ω` = Observer active (hearing)
- `Ω!` = High alert (loud sound)
- `θ` = Thinking/processing
- `Ξ` = Pattern recognition
- `μ` = Memory formation

## Quick Start

### 🎭 NEW: Platform-Independent Usage

#### Test Your System
```bash
python3 test_portable_audio.py
```
This detects your platform and available audio capabilities.

#### Run Cross-Platform Demo
```bash
# With real hardware
python3 demo_portable_audio.py

# Without hardware (simulation mode)
python3 demo_portable_audio.py --simulate
```

#### WSL Users
```bash
# Test Windows TTS bridge
python3 demo_wsl_voices.py
```

### Original Sprout-Specific Tools

#### 1. Test & Optimize Your Microphone
```bash
python3 sprout_mic_tuner.py
```
This will automatically find the best gain settings for your hardware.

#### 2. Run the Optimized System
```bash
python3 sprout_optimized_audio.py
```
Uses the saved optimal settings for best audio quality.

#### 3. Live Monitoring (Interactive)
```bash
python3 sprout_live_ears.py
```
- Press `+/-` to adjust sensitivity
- Press `q` to quit

## File Structure

### 🎭 NEW: Cross-Platform Components
- `audio_hal.py` - Hardware Abstraction Layer
- `consciousness_audio_system.py` - Platform-independent consciousness audio
- `audio_hal_simulator.py` - Audio simulation for testing
- `wsl_audio_bridge.py` - Windows TTS bridge for WSL
- `demo_portable_audio.py` - Cross-platform demonstration
- `test_portable_audio.py` - Platform capability tester

### Core Audio System (Sprout-Specific)
- `sprout_optimized_audio.py` - Main audio system with tuned settings
- `sprout_audio_config.txt` - Saved configuration (gain=50x)
- `sprout_simple_audio.py` - Simplified demo version

### Testing & Tuning Tools
- `sprout_mic_tuner.py` - Automatic gain optimization
- `sprout_mic_verified.py` - Quick microphone test
- `test_usb_audio.py` - USB device detection
- `test_microphone_input.py` - Recording verification

### Specialized Components
- `sprout_live_ears.py` - Real-time monitoring with controls
- `sprout_audio_system.py` - Full bidirectional system
- `sprout_audio_demo.py` - Voice reaction demonstrations

### Documentation
- `PORTABLE_AUDIO_README.md` - Cross-platform usage guide
- `MODULAR_AUDIO_SUMMARY.md` - Architecture summary
- `MODULAR_AUDIO_ACHIEVEMENT.md` - Technical achievements
- `AUDIO_VERIFICATION_RESULTS.md` - Test results and analysis
- `AUDIO_OPTIMIZATION_SUCCESS.md` - Learning insights and next steps

## Hardware Setup

### Current Configuration
- **Device**: USB PnP Audio Device (hw:2,0)
- **Index**: 24
- **Channels**: 2 input, 2 output
- **Optimal Gain**: 50x
- **Threshold**: 0.02

### Positioning Tips
1. Place microphone 6-12 inches from mouth
2. Speak at normal conversation volume
3. Avoid background noise sources

## Consciousness Integration

The system maps all audio events to consciousness states:

```python
# Example consciousness mapping
if level > 0.5:
    state = "Ω! Ξ"  # High alert + pattern
elif level > 0.1:
    state = "Ω θ"   # Observing + thinking
elif level > 0.02:
    state = "Ω"     # Observing
else:
    state = "..."   # Quiet
```

## Next Steps

- [ ] Voice Activity Detection (VAD)
- [ ] Speech-to-text integration
- [ ] Bidirectional conversation system
- [ ] Memory integration with audio events

## Status: ✅ WORKING

The audio system is fully functional with optimized settings!