#!/usr/bin/env python3
"""
Sprout's Microphone Tuner - Find the best settings for your mic
Helps adjust gain and sensitivity for optimal audio capture
"""

import pyaudio
import numpy as np
import subprocess
import time

def mic_tuner():
    """Interactive mic tuning tool"""
    print("🌱 Sprout's Microphone Tuner")
    print("=" * 50)
    print("Let's optimize your microphone settings!")
    print("=" * 50)
    
    subprocess.run(['espeak', '-s', '180', '-p', '75', '-v', 'en+f4',
                   'Let me help tune your microphone!'], capture_output=True)
    
    # USB device
    USB_DEVICE = 24
    p = pyaudio.PyAudio()
    
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    print("\n📊 Microphone Sensitivity Tips:")
    print("1. Position mic 6-12 inches from your mouth")
    print("2. Speak at normal conversation volume")
    print("3. Avoid background noise sources")
    print("4. Consider a USB mic with gain control")
    print("\n🎚️ Software Gain Adjustment:")
    
    # Different gain levels to test
    gain_levels = [1, 5, 10, 20, 50, 100]
    best_gain = 1
    best_peak = 0
    
    for gain in gain_levels:
        print(f"\n🔊 Testing gain level: {gain}x")
        print("Speak normally for 3 seconds...")
        time.sleep(1)
        
        subprocess.run(['espeak', '-s', '200', '-p', '90', '-v', 'en+f4',
                       f'Testing gain {gain}'], capture_output=True)
        
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=USB_DEVICE,
                frames_per_buffer=CHUNK
            )
            
            peaks = []
            start_time = time.time()
            
            while (time.time() - start_time) < 3:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Apply gain
                audio_data = audio_data.astype(np.float32) * gain
                audio_data = np.clip(audio_data, -32768, 32767)
                
                # Calculate level
                rms = np.sqrt(np.mean(audio_data**2))
                level = min(1.0, rms / 32768.0)
                peaks.append(level)
                
                # Visual feedback
                bar_length = int(level * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"\r[{bar}] {level:.3f}", end="", flush=True)
            
            stream.stop_stream()
            stream.close()
            
            avg_peak = np.mean(sorted(peaks)[-10:])  # Top 10 peaks average
            print(f"\n  Average peak: {avg_peak:.3f}")
            
            if 0.3 <= avg_peak <= 0.7:  # Good range
                print("  ✅ Good level!")
                if avg_peak > best_peak:
                    best_peak = avg_peak
                    best_gain = gain
            elif avg_peak < 0.3:
                print("  ⚠️  Too quiet")
            else:
                print("  ⚠️  Too loud (clipping)")
            
        except Exception as e:
            print(f"\nError: {e}")
    
    print(f"\n\n🎯 Recommended gain: {best_gain}x")
    print(f"   Best peak level: {best_peak:.3f}")
    
    # Save recommendation
    print("\n💾 Saving optimal settings...")
    with open("sprout_audio_config.txt", "w") as f:
        f.write(f"# Sprout Audio Configuration\n")
        f.write(f"MIC_GAIN={best_gain}\n")
        f.write(f"USB_DEVICE={USB_DEVICE}\n")
        f.write(f"THRESHOLD=0.02\n")
        f.write(f"# Tuned on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("✅ Configuration saved to sprout_audio_config.txt")
    
    # Hardware recommendations
    print("\n🎤 Hardware Tips for Better Audio:")
    print("1. USB microphones with built-in gain control:")
    print("   - Blue Yeti Nano ($50-80)")
    print("   - Samson Go Mic ($30-40)")
    print("   - Audio-Technica ATR2100x-USB ($60-80)")
    print("2. Position microphone on stable surface")
    print("3. Use pop filter or windscreen")
    print("4. Check USB power - use powered hub if needed")
    
    subprocess.run(['espeak', '-s', '160', '-p', '75', '-v', 'en+f4',
                   f'I recommend using gain {best_gain} for your setup!'], 
                   capture_output=True)
    
    p.terminate()

if __name__ == "__main__":
    mic_tuner()