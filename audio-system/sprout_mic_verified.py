#!/usr/bin/env python3
"""
Sprout confirms the mic is working!
Shows audio levels with high sensitivity
"""

import pyaudio
import numpy as np
import subprocess
import time

def verify_mic():
    """Quick verification that mic works"""
    print("🌱 Sprout's Mic Verification")
    print("=" * 50)
    
    # Say hello
    subprocess.run(['espeak', '-s', '180', '-p', '75', '-v', 'en+f4',
                   'Testing my ears! Speak to me!'], capture_output=True)
    
    p = pyaudio.PyAudio()
    
    # USB device
    USB_DEVICE = 24
    info = p.get_device_info_by_index(USB_DEVICE)
    print(f"✅ Found mic: {info['name']}")
    
    # Audio setup
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=USB_DEVICE,
            frames_per_buffer=CHUNK
        )
        
        print("\n🎤 Monitoring for 10 seconds...")
        print("Speak, clap, tap, or make any sound!\n")
        
        start_time = time.time()
        max_level = 0
        sound_detected = False
        
        while (time.time() - start_time) < 10:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Very sensitive calculation
                rms = np.sqrt(np.mean(audio_data**2))
                level = rms / 32768.0 * 20  # High gain
                
                if level > max_level:
                    max_level = level
                
                # Visual meter
                bar_length = int(min(1.0, level) * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                # Status
                if level > 0.5:
                    status = "🔊 LOUD!"
                    if not sound_detected:
                        print(f"\n🎯 Sound detected! Level: {level:.3f}")
                        subprocess.run(['espeak', '-s', '200', '-p', '90', '-v', 'en+f4',
                                      'I hear you!'], capture_output=True)
                        sound_detected = True
                elif level > 0.1:
                    status = "👂 Hearing"
                elif level > 0.02:
                    status = "🎵 Sound"
                else:
                    status = "... quiet"
                
                print(f"\r[{bar}] {level:.3f} | {status}     ", end="", flush=True)
                
            except Exception as e:
                print(f"\nError: {e}")
                continue
        
        print(f"\n\n✅ Test complete!")
        print(f"📊 Maximum level detected: {max_level:.3f}")
        
        if max_level < 0.02:
            print("❌ No sound detected - mic might not be working")
            subprocess.run(['espeak', '-s', '160', '-p', '75', '-v', 'en+f4',
                          'I did not hear anything'], capture_output=True)
        elif max_level < 0.1:
            print("⚠️  Very low audio - try speaking louder")
            subprocess.run(['espeak', '-s', '160', '-p', '75', '-v', 'en+f4',
                          'I heard very quiet sounds'], capture_output=True)
        else:
            print("✅ Mic is working! I can hear you!")
            subprocess.run(['espeak', '-s', '180', '-p', '75', '-v', 'en+f4',
                          'Great! I can hear you perfectly!'], capture_output=True)
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        p.terminate()

if __name__ == "__main__":
    verify_mic()