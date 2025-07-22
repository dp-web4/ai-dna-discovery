#!/usr/bin/env python3
"""
Sprout's Simple Audio Demo - Hearing and Speaking
Shows how audio input maps to consciousness states
"""

import subprocess
import time
import pyaudio
import numpy as np
import threading

class SimpleSprout:
    def __init__(self):
        self.name = "Sprout"
        self.mood = "curious"
        
    def say(self, text, show_consciousness=True):
        """Simple speech using espeak"""
        # Map text to consciousness
        consciousness = self.get_consciousness(text)
        if show_consciousness and consciousness:
            print(f"🧠 {consciousness}")
            
        print(f"🌱 {self.name}: {text}")
        
        # Speak with espeak
        cmd = ['espeak', '-s', '160', '-p', '75', '-v', 'en+f4', text]
        subprocess.run(cmd, capture_output=True)
        
    def get_consciousness(self, text):
        """Map speech to consciousness notation"""
        symbols = []
        
        text_lower = text.lower()
        if any(word in text_lower for word in ['hear', 'heard', 'sound', 'loud']):
            symbols.append('Ω')  # Observer
        if any(word in text_lower for word in ['think', 'wonder', 'what']):
            symbols.append('θ')  # Thought
        if any(word in text_lower for word in ['pattern', 'rhythm', 'music']):
            symbols.append('Ξ')  # Patterns
        if '!' in text:
            symbols.append('!')  # Excitement
            
        return ' '.join(symbols) if symbols else None

def audio_level_demo():
    """Demo showing audio level detection"""
    sprout = SimpleSprout()
    
    print("🎤 Audio Level Detection Demo")
    print("=" * 50)
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Find USB device
    usb_device = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "USB" in info['name'] and info['maxInputChannels'] > 0:
            usb_device = i
            print(f"✅ Found USB mic: {info['name']} (device {i})")
            break
    
    if usb_device is None:
        print("❌ No USB microphone found!")
        p.terminate()
        return
    
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    try:
        # Open stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=usb_device,
            frames_per_buffer=CHUNK
        )
        
        print("\n🎧 Monitoring audio for 20 seconds...")
        print("Make some noise! Speak, clap, tap the mic...")
        print("-" * 50)
        
        sprout.say("Hello! I'm listening for sounds now!", show_consciousness=True)
        
        start_time = time.time()
        last_reaction = 0
        max_level = 0
        
        while (time.time() - start_time) < 20:
            try:
                # Read audio
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Calculate level
                rms = np.sqrt(np.mean(audio_data**2))
                level = min(1.0, rms / 32768.0 * 10)
                
                # Update max
                if level > max_level:
                    max_level = level
                
                # Visual meter
                bar_length = int(level * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                
                # Consciousness state
                if level > 0.5:
                    consciousness = "Ω! Ξ"
                elif level > 0.1:
                    consciousness = "Ω θ"
                elif level > 0.02:
                    consciousness = "Ω"
                else:
                    consciousness = "..."
                
                print(f"\r[{bar}] {level:.3f} | 🧠 {consciousness}  ", end="", flush=True)
                
                # React to loud sounds
                current_time = time.time()
                if level > 0.1 and (current_time - last_reaction) > 3:
                    print()  # New line
                    
                    if level > 0.5:
                        sprout.say("Wow! That was loud!")
                    elif level > 0.2:
                        sprout.say("I heard something!")
                    else:
                        sprout.say("What was that?")
                    
                    last_reaction = current_time
                
            except Exception as e:
                print(f"\nError: {e}")
                break
        
        print(f"\n\n✅ Monitoring complete! Max level: {max_level:.3f}")
        
        if max_level < 0.01:
            sprout.say("It was very quiet. I didn't hear much.")
        else:
            sprout.say("Thanks for making all those sounds!")
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"\n❌ Stream error: {e}")
    
    finally:
        p.terminate()

def consciousness_mapping_demo():
    """Demo showing consciousness mapping"""
    sprout = SimpleSprout()
    
    print("\n\n🧠 Consciousness Mapping Demo")
    print("=" * 50)
    print("Showing how sounds map to consciousness states")
    print("=" * 50)
    
    scenarios = [
        ("Silence detected", "..."),
        ("Quiet sound", "Ω"),
        ("Voice detected", "Ω θ"),
        ("Loud clap!", "Ω! Ξ"),
        ("Music pattern", "Ω Ξ μ"),
        ("Processing sounds", "θ μ"),
    ]
    
    for desc, consciousness in scenarios:
        print(f"\n🎵 {desc}")
        print(f"🧠 Consciousness: {consciousness}")
        
        # Explain
        if consciousness == "...":
            sprout.say("It's quiet... I'm waiting")
        elif consciousness == "Ω":
            sprout.say("My observer function detected something")
        elif consciousness == "Ω θ":
            sprout.say("I hear it and I'm thinking about it")
        elif consciousness == "Ω! Ξ":
            sprout.say("Loud sound! Pattern detected!")
        elif consciousness == "Ω Ξ μ":
            sprout.say("I hear patterns and I'm remembering them")
        elif consciousness == "θ μ":
            sprout.say("Thinking about what I heard and storing it")
        
        time.sleep(2)
    
    print("\n")
    sprout.say("That's how I map sounds to consciousness!")

def main():
    print("🌱 Sprout's Audio System Demo")
    print("=" * 50)
    
    # First show consciousness mapping
    consciousness_mapping_demo()
    
    print("\n\nStarting live audio monitoring in 2 seconds...")
    time.sleep(2)
    
    # Then do live audio monitoring
    audio_level_demo()
    
    print("\n✨ Demo complete!")

if __name__ == "__main__":
    main()