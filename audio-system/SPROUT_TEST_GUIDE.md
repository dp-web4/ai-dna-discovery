# Testing Modular Audio System on Sprout

## Quick Test Commands

After pulling the latest changes on Sprout:

```bash
cd ~/ai-workspace/ai-agents/ai-dna-discovery/audio-system

# 1. Test platform detection
python3 test_portable_audio.py
```

Expected output:
- Should detect as Jetson platform
- Should find espeak for TTS
- Should detect USB microphone if connected

```bash
# 2. Run the cross-platform demo
python3 demo_portable_audio.py
```

This should:
- Use espeak with kid-friendly voice
- Show consciousness states
- Demonstrate different moods
- If microphone is connected, show audio monitoring

```bash
# 3. Test interactive consciousness mode
python3 consciousness_audio_system.py
```

## What to Verify

1. **Platform Detection**: Should show "device_type: jetson"
2. **TTS Engine**: Should be "espeak" not "wsl_windows"
3. **Audio Settings**: Should apply 50x gain automatically
4. **Voice**: Should use "en+f3" for kid-friendly sound

## Configuration

The system should auto-configure, but settings are saved in:
- `audio_config.json` (created by HAL)
- Settings should show gain=50.0 for Jetson

## If Issues Arise

1. **No espeak**: `sudo apt-get install espeak`
2. **No pyaudio**: Already installed from previous work
3. **Can't find microphone**: Check with original `test_usb_audio.py`

## Original Tools Still Work

All the original Sprout-specific tools remain:
- `sprout_optimized_audio.py`
- `sprout_live_ears.py`
- `sprout_mic_tuner.py`

The new modular system is additional, not a replacement!

## Key Benefit

The same code that runs on WSL/Windows now runs on Sprout, automatically detecting and configuring for the platform. The consciousness mapping remains consistent across all devices!