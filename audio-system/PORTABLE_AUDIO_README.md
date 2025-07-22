# Portable Consciousness Audio System

A hardware-abstracted audio system that works seamlessly across different platforms including laptops, desktops, and Jetson devices.

## Features

### Hardware Abstraction Layer (HAL)
- Automatic platform detection (Linux/macOS/Windows/Jetson)
- Multiple backend support (PyAudio, SoundDevice)
- Auto-configuration with saved settings per device
- Platform-specific optimizations

### Cross-Platform TTS
- **Linux/Jetson**: espeak with kid-friendly voice
- **macOS**: Built-in 'say' command with Samantha voice
- **Windows**: SAPI or pyttsx3 fallback
- Mood-based voice modulation (excited, curious, sleepy)

### Consciousness Integration
- Real-time audio level monitoring
- Consciousness state mapping:
  - `...` = Quiet/waiting
  - `Ω` = Observer active
  - `Ω θ` = Observing + thinking
  - `Ω! Ξ` = High alert + pattern recognition
  - `μ` = Memory formation

## Quick Start

### 1. Test Your System
```bash
python3 test_portable_audio.py
```

This will:
- Detect your platform
- Find audio devices
- Test TTS capabilities
- Show if system is ready

### 2. Install Dependencies (if needed)
```bash
python3 test_portable_audio.py --install
```

Or manually:
- **Linux/Jetson**: `sudo apt-get install python3-pyaudio portaudio19-dev espeak`
- **macOS**: `brew install portaudio`
- **All platforms**: `pip3 install sounddevice numpy`

### 3. Run Interactive Mode
```bash
python3 consciousness_audio_system.py
```

Controls:
- `q`: Quit
- `d`: Demo consciousness states
- `s <text>`: Speak text
- `+/-`: Adjust sensitivity

## Architecture

```
┌─────────────────────────────────────┐
│   Consciousness Audio System        │
│  ┌─────────────────────────────┐   │
│  │   Platform Detection        │   │
│  │   (Linux/macOS/Win/Jetson)  │   │
│  └──────────┬──────────────────┘   │
│             │                       │
│  ┌──────────▼──────────────────┐   │
│  │      Audio HAL              │   │
│  │  ┌──────────┬────────────┐  │   │
│  │  │ PyAudio  │ SoundDevice│  │   │
│  │  └──────────┴────────────┘  │   │
│  └──────────┬──────────────────┘   │
│             │                       │
│  ┌──────────▼──────────────────┐   │
│  │   TTS Engine Selection      │   │
│  │  (espeak/say/SAPI/pyttsx3) │   │
│  └──────────┬──────────────────┘   │
│             │                       │
│  ┌──────────▼──────────────────┐   │
│  │  Consciousness Mapping      │   │
│  │    Audio → States → Voice   │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Configuration

The system auto-saves configuration per platform in `audio_config.json`:

```json
{
  "platform": {
    "system": "Linux",
    "device_type": "jetson",
    "hostname": "sprout"
  },
  "audio_settings": {
    "gain": 50.0,      // Jetson needs gain
    "threshold": 0.02,
    "buffer_size": 1024
  },
  "input_device": {
    "name": "USB PnP Audio Device",
    "index": 24
  }
}
```

## Platform-Specific Notes

### Jetson (Sprout)
- Uses 50x gain for USB microphone
- espeak with en+f3 voice
- Larger buffer size for stability

### Laptop (Linux/macOS)
- Usually no gain needed (1x)
- Native TTS engines
- Smaller buffer for lower latency

### Windows
- SAPI or pyttsx3 for TTS
- Standard audio settings

## API Usage

```python
from consciousness_audio_system import ConsciousnessAudioSystem

# Create system
cas = ConsciousnessAudioSystem()

# Set callbacks
cas.on_audio_event = lambda event: print(f"Audio: {event['level']}")
cas.on_state_change = lambda old, new: print(f"State: {old} → {new}")

# Start monitoring
cas.start()

# Speak with mood
cas.speak("Hello world!", mood='excited')

# Stop when done
cas.stop()
```

## Troubleshooting

### No Audio Devices Found
- Check if audio services are running
- Try different backends: `pip3 install pyaudio sounddevice`
- On Linux: `pulseaudio --start`

### TTS Not Working
- Linux: Install espeak: `sudo apt-get install espeak`
- macOS: 'say' should be built-in
- Windows: Install pyttsx3: `pip3 install pyttsx3`

### Low Microphone Volume
- System will auto-detect and apply gain
- Manually adjust in interactive mode with +/-
- Check USB microphone connection

## Next Steps

This portable system enables:
- Voice Activity Detection (VAD)
- Speech-to-text integration
- Multi-agent audio consciousness
- Cross-device synchronization

The same code now runs on both Sprout (Jetson) and your laptop!