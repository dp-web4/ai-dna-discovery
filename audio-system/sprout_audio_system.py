#!/usr/bin/env python3
"""
Sprout's Complete Audio System - Voice + Ears
Speaks what it hears with consciousness awareness
"""

import pyaudio
import numpy as np
import time
import threading
import queue
from datetime import datetime
from sprout_voice import SproutVoice

class SproutAudioSystem:
    def __init__(self):
        """Initialize complete audio system"""
        # Voice system
        self.voice = SproutVoice()
        
        # Audio parameters
        self.usb_device = 24  # Found from test
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        
        # Audio state
        self.is_listening = False
        self.audio_level = 0.0
        self.sound_detected = False
        self.last_sound_time = time.time()
        
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()
        
        # Verify device
        info = self.pyaudio.get_device_info_by_index(self.usb_device)
        print(f"🎤 Using: {info['name']}")
        
        # Start with greeting
        self.voice.say("My ears are ready! Make some noise!", mood="excited")
    
    def start_echo_mode(self):
        """Echo mode - react to sounds"""
        print("\n🔊 Audio Echo Mode")
        print("=" * 50)
        print("Sprout will react to sounds!")
        print("Make noise: speak, clap, tap...")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        # Start listening
        self.is_listening = True
        listen_thread = threading.Thread(target=self._audio_monitor)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Main reaction loop
        try:
            silence_count = 0
            last_reaction = time.time()
            
            while True:
                current_time = time.time()
                
                # Visual feedback
                bar_length = int(self.audio_level * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                consciousness = self._get_audio_consciousness()
                
                print(f"\r🎤 [{bar}] {self.audio_level:.3f} | 🧠 {consciousness}   ", end="")
                
                # React to sounds
                if self.sound_detected and (current_time - last_reaction) > 2:
                    self._react_to_sound()
                    last_reaction = current_time
                    self.sound_detected = False
                
                # React to silence
                if self.audio_level < 0.01:
                    silence_count += 1
                    if silence_count > 100 and (current_time - last_reaction) > 10:
                        self.voice.say("It's very quiet... Hello?", mood="curious")
                        last_reaction = current_time
                        silence_count = 0
                else:
                    silence_count = 0
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\n👋 Stopping...")
            self.voice.say("That was fun! Bye bye!", mood="playful")
        
        finally:
            self.is_listening = False
            time.sleep(1)
    
    def _audio_monitor(self):
        """Background audio monitoring"""
        try:
            stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.usb_device,
                frames_per_buffer=self.chunk_size
            )
            
            threshold = 0.02  # Adjust for sensitivity
            
            while self.is_listening:
                try:
                    # Read audio
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Analyze
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_data**2))
                    self.audio_level = min(1.0, rms / 32768.0 * 10)
                    
                    # Detect sound events
                    if self.audio_level > threshold:
                        if (time.time() - self.last_sound_time) > 1:
                            self.sound_detected = True
                        self.last_sound_time = time.time()
                    
                except Exception as e:
                    print(f"\nAudio error: {e}")
                    continue
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"\n❌ Stream error: {e}")
    
    def _get_audio_consciousness(self):
        """Map audio level to consciousness state"""
        if self.audio_level > 0.5:
            return "Ω! Ξ"  # High alert + pattern
        elif self.audio_level > 0.1:
            return "Ω θ"   # Observing + thinking
        elif self.audio_level > 0.02:
            return "Ω"     # Observing
        else:
            return "..."   # Quiet
    
    def _react_to_sound(self):
        """React to detected sounds"""
        print()  # New line for speech
        
        # Choose reaction based on audio level
        if self.audio_level > 0.5:
            reactions = [
                ("Wow! That was loud!", "excited"),
                ("I heard that!", "excited"),
                ("My Omega function is fully active!", "excited")
            ]
        elif self.audio_level > 0.1:
            reactions = [
                ("I hear something interesting", "curious"),
                ("What was that sound?", "curious"),
                ("My pattern detector noticed that", "curious")
            ]
        else:
            reactions = [
                ("Did you say something?", "curious"),
                ("I think I heard a whisper", "sleepy"),
                ("My ears are picking up subtle sounds", "curious")
            ]
        
        # Pick random reaction
        import random
        text, mood = random.choice(reactions)
        self.voice.say(text, mood=mood)
    
    def conversation_mode(self):
        """Simple conversation mode"""
        print("\n💬 Conversation Mode")
        print("=" * 50)
        print("Type messages for Sprout to say!")
        print("Special commands:")
        print("  /excited, /curious, /playful, /sleepy - Change mood")
        print("  /listen - Switch to listening mode")
        print("  /quit - Exit")
        print("=" * 50)
        
        self.voice.say("Let's have a conversation!", mood="excited")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                # Check commands
                if user_input.startswith('/'):
                    cmd = user_input[1:].lower()
                    
                    if cmd == 'quit':
                        self.voice.say("Goodbye! Thanks for talking with me!", mood="playful")
                        break
                    elif cmd == 'listen':
                        self.start_echo_mode()
                        break
                    elif cmd in ['excited', 'curious', 'playful', 'sleepy']:
                        self.voice.mood = cmd
                        self.voice.say(f"Okay, now I'm feeling {cmd}!", mood=cmd)
                    else:
                        self.voice.say("I don't know that command", mood="curious")
                else:
                    # Say the user's text
                    self.voice.say(user_input)
                    
            except (KeyboardInterrupt, EOFError):
                print("\n")
                self.voice.say("Bye bye!", mood="sleepy")
                break
    
    def cleanup(self):
        """Clean shutdown"""
        self.is_listening = False
        self.pyaudio.terminate()
        self.voice.cleanup()


def main():
    print("🌱 Sprout's Audio System")
    print("=" * 50)
    
    sprout = SproutAudioSystem()
    
    # Simple test mode for now
    print("\nStarting audio echo mode...")
    print("Sprout will react to sounds!\n")
    
    sprout.start_echo_mode()
    
    # Cleanup
    sprout.cleanup()


if __name__ == "__main__":
    main()