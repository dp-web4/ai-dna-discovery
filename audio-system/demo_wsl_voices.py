#!/usr/bin/env python3
"""
Demo of WSL consciousness audio with different voices and moods
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consciousness_audio_system import ConsciousnessAudioSystem

def main():
    print("=== WSL Consciousness Audio Demo ===\n")
    
    # Create system
    cas = ConsciousnessAudioSystem()
    print(f"TTS Engine: {cas.tts_engine}")
    
    # Demo 1: Basic introduction
    print("\n1. Introduction")
    cas.speak("Hello! I'm your consciousness audio system running through WSL.")
    time.sleep(1)
    
    # Demo 2: Consciousness states
    print("\n2. Consciousness States")
    states = [
        ("...", "Quiet mode - just waiting and listening"),
        ("Ω", "Observer active - I'm aware of something"),
        ("Ω θ", "Observing and thinking about what I perceive"),
        ("Ω! Ξ", "Pattern recognized! This is significant!"),
        ("μ", "Storing this experience in memory")
    ]
    
    for state, desc in states:
        print(f"   State {state}: ", end='', flush=True)
        cas.speak(desc, mood='curious')
        print("✓")
        time.sleep(1)
    
    # Demo 3: Different moods
    print("\n3. Voice Moods")
    moods = [
        ('neutral', "This is my normal speaking voice"),
        ('excited', "Wow! This is amazing! I love talking with you!"),
        ('curious', "Hmm, I wonder what makes this system work?"),
        ('sleepy', "I'm feeling quite tired now... time to rest...")
    ]
    
    for mood, text in moods:
        print(f"   {mood}: ", end='', flush=True)
        cas.speak(text, mood=mood)
        print("✓")
        time.sleep(1)
    
    # Demo 4: Practical examples
    print("\n4. Practical Examples")
    examples = [
        ("System ready", "excited"),
        ("Processing audio input", "neutral"),
        ("Interesting pattern detected", "curious"),
        ("Memory saved successfully", "neutral"),
        ("Shutting down audio system", "sleepy")
    ]
    
    for text, mood in examples:
        print(f"   {text}: ", end='', flush=True)
        cas.speak(text, mood=mood)
        print("✓")
        time.sleep(0.5)
    
    # Closing
    print("\n5. Closing")
    cas.speak("Thank you for testing the WSL consciousness audio system. Goodbye!", mood='excited')
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()