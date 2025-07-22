#!/usr/bin/env python3
"""
Test if Sprout can actually hear YOU!
Records audio and plays it back to verify the mic works.
"""

import pyaudio
import wave
import numpy as np
import time
import subprocess

def test_mic_input():
    """Test microphone with recording and playback"""
    print("🎤 Microphone Input Test")
    print("=" * 50)
    
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    USB_DEVICE = 24  # Our USB mic
    
    p = pyaudio.PyAudio()
    
    # Verify device
    info = p.get_device_info_by_index(USB_DEVICE)
    print(f"Using: {info['name']}")
    print(f"Channels: {info['maxInputChannels']} input")
    
    print("\n📢 INSTRUCTIONS:")
    print("1. I'll count down from 3")
    print("2. When I say 'GO!', speak clearly into the mic")
    print("3. Say something like: 'Hello Sprout, can you hear me?'")
    print("4. I'll record for 5 seconds")
    print("5. Then I'll play it back so you can verify")
    print("\nReady? Starting in 2 seconds...")
    time.sleep(2)
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        subprocess.run(['espeak', '-s', '160', '-p', '75', f'{i}'], capture_output=True)
        time.sleep(1)
    
    print("🔴 GO! SPEAK NOW!")
    subprocess.run(['espeak', '-s', '200', '-p', '90', 'Go! Speak now!'], capture_output=True)
    
    # Start recording
    frames = []
    max_level = 0
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=USB_DEVICE,
            frames_per_buffer=CHUNK
        )
        
        print("\n🔴 RECORDING...")
        start_time = time.time()
        
        while (time.time() - start_time) < RECORD_SECONDS:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Show level meter
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data**2))
            level = min(1.0, rms / 32768.0 * 10)
            max_level = max(max_level, level)
            
            bar_length = int(level * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"\r[{bar}] {level:.3f}", end="", flush=True)
        
        print("\n\n✅ Recording complete!")
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"\n❌ Recording error: {e}")
        p.terminate()
        return
    
    # Save recording
    filename = "sprout_mic_test.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    print(f"💾 Saved to {filename}")
    
    # Analyze what we recorded
    print(f"\n📊 Analysis:")
    print(f"  Max audio level: {max_level:.3f}")
    
    if max_level < 0.01:
        print("  ❌ VERY LOW AUDIO - Mic might not be working!")
        print("  Try: Speaking louder, checking connections")
    elif max_level < 0.05:
        print("  ⚠️  Low audio levels - Speak louder/closer")
    else:
        print("  ✅ Good audio levels detected!")
    
    # Playback
    print("\n🔊 Playing back your recording...")
    subprocess.run(['espeak', '-s', '160', '-p', '75', 'Playing back your recording'], capture_output=True)
    time.sleep(1)
    
    # Play the WAV file
    try:
        subprocess.run(['aplay', '-D', 'hw:2,0', filename], capture_output=True)
    except:
        # Fallback to default device
        subprocess.run(['aplay', filename], capture_output=True)
    
    print("\n🎯 Did you hear your voice?")
    print("  - YES: Great! The mic is working!")
    print("  - NO: Let's troubleshoot...")
    print("  - STATIC/NOISE: Mic might be too sensitive")
    
    p.terminate()
    
    # Sprout's reaction
    time.sleep(1)
    subprocess.run(['espeak', '-s', '160', '-p', '75', '-v', 'en+f4', 
                   'Did I hear you correctly? I hope so!'], capture_output=True)

def live_monitor_test():
    """Live monitoring with immediate feedback"""
    print("\n\n🎧 Live Monitoring Test")
    print("=" * 50)
    print("Speak and watch the meter respond in real-time!")
    print("I'll react when I detect sound")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    CHUNK = 512  # Smaller for more responsive
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    USB_DEVICE = 24
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=USB_DEVICE,
            frames_per_buffer=CHUNK
        )
        
        subprocess.run(['espeak', '-s', '180', '-p', '75', '-v', 'en+f4',
                       'Listening! Make some noise!'], capture_output=True)
        
        last_reaction = 0
        sound_detected = False
        
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data**2))
                level = min(1.0, rms / 32768.0 * 10)
                
                # Visual meter
                bar_length = int(level * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                
                # Consciousness state
                if level > 0.1:
                    state = "Ω! HEARING YOU!"
                    if not sound_detected:
                        sound_detected = True
                        print(f"\n🎯 SOUND DETECTED! Level: {level:.3f}")
                elif level > 0.02:
                    state = "Ω listening..."
                else:
                    state = "... quiet"
                    sound_detected = False
                
                print(f"\r[{bar}] {level:.3f} | {state}     ", end="", flush=True)
                
                # Voice feedback for loud sounds
                current_time = time.time()
                if level > 0.1 and (current_time - last_reaction) > 3:
                    print()
                    subprocess.run(['espeak', '-s', '200', '-p', '90', '-v', 'en+f4',
                                  'I hear you!'], capture_output=True)
                    last_reaction = current_time
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"\n❌ Stream error: {e}")
    
    finally:
        p.terminate()
        print("\n\n✅ Monitoring stopped")

def main():
    print("🌱 Sprout's Microphone Verification Test")
    print("=" * 50)
    print("Let's make sure I can actually hear you!")
    print("=" * 50)
    
    # First do recording test
    test_mic_input()
    
    # Then offer live monitoring
    print("\n\nSkipping live monitoring test in non-interactive mode.")
    
    print("\n✨ Test complete!")
    subprocess.run(['espeak', '-s', '160', '-p', '75', '-v', 'en+f4',
                   'Thanks for testing my ears!'], capture_output=True)

if __name__ == "__main__":
    main()