#!/usr/bin/env python3
"""
Quick voice level test - show exactly what energy your voice produces
"""

import pyaudio
import numpy as np
import time

USB_DEVICE = 24
GAIN = 50.0

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=44100,
    input=True,
    input_device_index=USB_DEVICE,
    frames_per_buffer=4096
)

print("🎤 Say something NOW - showing your actual voice energy levels:")
print("Background should be ~0.05, speech should be >0.1")
print("Press Ctrl+C to stop\n")

try:
    max_seen = 0
    for i in range(200):  # ~8 seconds
        audio_data = stream.read(4096, exception_on_overflow=False)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_np = audio_np / 32768.0 * GAIN
        
        # Downsample like we do in the main system
        if len(audio_np) >= 3:
            downsampled = audio_np[::3][:1024]
            if len(downsampled) == 1024:
                energy = np.sqrt(np.mean(downsampled ** 2))
                max_seen = max(max_seen, energy)
                
                if energy > 0.08:  # Show when above background
                    print(f"🔊 VOICE: {energy:.3f} (max so far: {max_seen:.3f})")
                elif i % 20 == 0:  # Show background every ~0.8s
                    print(f"💤 Background: {energy:.3f} (max so far: {max_seen:.3f})")
        
        time.sleep(0.04)

except KeyboardInterrupt:
    print(f"\n✅ Your voice peaks at: {max_seen:.3f}")
    if max_seen > 0.1:
        print(f"💡 Use energy_threshold: {max_seen * 0.3:.3f}")
    else:
        print("❌ Speak louder or closer to mic!")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()