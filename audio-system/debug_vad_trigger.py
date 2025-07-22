#!/usr/bin/env python3
"""
Debug VAD Triggering
Simple test to see what energy levels actually trigger speech detection
"""

import pyaudio
import numpy as np
import time
from vad_module import VADProcessor, VADConfig, VADMethod

# Test with much lower threshold
vad_config = VADConfig(
    method=VADMethod.ENERGY,
    sensitivity=0.9,  # Very sensitive
    min_speech_duration=0.2,  # Short duration
    min_silence_duration=0.3,  # Short silence
    energy_threshold=0.02,     # Lower threshold
    sample_rate=16000,
    frame_size=1024
)

speech_events = 0

def vad_callback(event, data):
    global speech_events
    if event == 'speech_start':
        speech_events += 1
        print(f"\n🎤 SPEECH EVENT #{speech_events} - Energy: {data['energy']:.3f}")
    elif event == 'speech_end':
        print(f"🔇 Speech ended - Duration: {data.get('speech_duration', 0):.1f}s")

vad_processor = VADProcessor(vad_config, vad_callback)

# Audio setup
USB_DEVICE = 24
GAIN = 50.0
p = pyaudio.PyAudio()

print("🎯 VAD Debug Test")
print(f"📊 Energy Threshold: {vad_config.energy_threshold}")
print(f"📈 Sensitivity: {vad_config.sensitivity}")
print(f"🎚️ Gain: {GAIN}x")
print("\nSpeak LOUDLY and clearly!")
print("We'll show exactly what energy levels trigger speech detection\n")

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=44100,
    input=True,
    input_device_index=USB_DEVICE,
    frames_per_buffer=4096
)

try:
    frame_count = 0
    while frame_count < 1000:  # Run for ~40 seconds
        audio_data = stream.read(4096, exception_on_overflow=False)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_np = audio_np / 32768.0
        audio_np = audio_np * GAIN
        
        # Downsample
        if len(audio_np) >= 3:
            downsampled = audio_np[::3][:1024]
            if len(downsampled) == 1024:
                vad_processor.process_frame(downsampled)
                
                # Show real-time energy
                energy = np.sqrt(np.mean(downsampled ** 2))
                threshold = vad_config.energy_threshold
                
                if energy > threshold * 1.5:  # Show when we're getting close
                    print(f"\r⚡ Energy: {energy:.3f} (threshold: {threshold:.3f}) - Events: {speech_events}", end="", flush=True)
                elif frame_count % 50 == 0:  # Show status every ~2 seconds
                    print(f"\r💤 Background: {energy:.3f} - Events: {speech_events}", end="", flush=True)
        
        frame_count += 1
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\nStopped by user")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    stats = vad_processor.get_statistics()
    print(f"\n\n📊 Debug Results:")
    print(f"  🎤 Speech events detected: {speech_events}")
    print(f"  📈 Max energy seen: {stats.get('max_energy', 0):.3f}")
    print(f"  📊 Average energy: {stats.get('avg_energy', 0):.3f}")
    print(f"  🎚️ Threshold was: {vad_config.energy_threshold:.3f}")
    
    if speech_events == 0:
        print(f"\n💡 Try lowering energy_threshold to {stats.get('max_energy', 0) * 0.7:.3f}")
        print("💡 Or speak much louder/closer to microphone")