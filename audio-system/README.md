# Sprout's Audio System 🌱🎤🔊

Complete audio system for Sprout (Jetson Orin Nano) with consciousness-aware voice and hearing capabilities.

## System Overview

### Voice Output (TTS)
- Kid-friendly voice using espeak
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

### 1. Test & Optimize Your Microphone
```bash
python3 sprout_mic_tuner.py
```
This will automatically find the best gain settings for your hardware.

### 2. Run the Optimized System
```bash
python3 sprout_optimized_audio.py
```
Uses the saved optimal settings for best audio quality.

### 3. Live Monitoring (Interactive)
```bash
python3 sprout_live_ears.py
```
- Press `+/-` to adjust sensitivity
- Press `q` to quit

## File Structure

### Core Audio System
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