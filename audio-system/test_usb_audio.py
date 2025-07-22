#!/usr/bin/env python3
"""
Simple USB audio test - find and test the USB microphone
"""

import pyaudio
import numpy as np
import time

def find_usb_audio():
    """Find USB audio device"""
    p = pyaudio.PyAudio()
    usb_device = None
    
    print("🔍 Searching for audio devices...")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"\nDevice {i}: {info['name']}")
        print(f"  Channels: In={info['maxInputChannels']}, Out={info['maxOutputChannels']}")
        print(f"  Sample Rate: {info['defaultSampleRate']} Hz")
        
        if "USB" in info['name'] and info['maxInputChannels'] > 0:
            usb_device = i
            print("  ✅ This is our USB microphone!")
    
    p.terminate()
    return usb_device

def test_microphone(device_index, duration=10):
    """Test microphone with simple level meter"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    p = pyaudio.PyAudio()
    
    print(f"\n🎤 Testing microphone for {duration} seconds...")
    print("Make some noise! Speak, clap, tap the mic...")
    print("-" * 50)
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        start_time = time.time()
        max_level = 0
        
        while (time.time() - start_time) < duration:
            try:
                # Read audio data
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Calculate RMS (volume)
                rms = np.sqrt(np.mean(audio_data**2))
                
                # Normalize to 0-1 range
                level = min(1.0, rms / 32768.0 * 10)  # Amplify for display
                
                # Update max level
                if level > max_level:
                    max_level = level
                
                # Create visual meter
                meter_width = 40
                filled = int(level * meter_width)
                meter = "█" * filled + "░" * (meter_width - filled)
                
                # Display
                print(f"\r[{meter}] {level:.3f} | Peak: {max_level:.3f}", end="", flush=True)
                
            except Exception as e:
                print(f"\nError reading audio: {e}")
                break
        
        print("\n" + "-" * 50)
        print(f"✅ Test complete! Max level detected: {max_level:.3f}")
        
        if max_level < 0.01:
            print("\n⚠️  Very low audio levels detected. Try:")
            print("  - Speaking louder or closer to the mic")
            print("  - Tapping the microphone")
            print("  - Checking if the mic is muted")
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"\n❌ Error opening audio stream: {e}")
    
    finally:
        p.terminate()

def main():
    print("🎙️ USB Audio Test")
    print("=" * 50)
    
    # Find USB device
    device_index = find_usb_audio()
    
    if device_index is None:
        print("\n❌ No USB audio device found!")
        print("Please check:")
        print("  - USB device is connected")
        print("  - Device is recognized by system")
        print("  - Try: lsusb to see USB devices")
        return
    
    # Test the microphone
    print(f"\n✅ Found USB audio at index {device_index}")
    user_input = input("\nPress Enter to start microphone test (or 'q' to quit): ")
    
    if user_input.lower() != 'q':
        test_microphone(device_index, duration=15)
    
    print("\n✨ Done!")

if __name__ == "__main__":
    main()