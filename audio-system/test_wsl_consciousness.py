#!/usr/bin/env python3
"""
Simple interactive test for WSL consciousness audio
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consciousness_audio_system import ConsciousnessAudioSystem
import time


def main():
    print("=== WSL Consciousness Audio Test ===\n")
    
    # Create consciousness audio system
    cas = ConsciousnessAudioSystem()
    
    print(f"TTS Engine: {cas.tts_engine}")
    print(f"Platform: WSL on {cas.hal.platform['system']}\n")
    
    # Interactive loop
    print("Enter text to speak, or commands:")
    print("  /excited <text> - speak with excitement")
    print("  /curious <text> - speak with curiosity")
    print("  /sleepy <text> - speak sleepily")
    print("  /state - demonstrate consciousness states")
    print("  /test - quick test of all moods")
    print("  /quit - exit\n")
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == '/quit':
                cas.speak("Goodbye! It was nice talking with you!", mood='sleepy')
                break
                
            elif user_input.lower() == '/state':
                print("\nDemonstrating consciousness states...")
                states = [
                    ("...", "I'm in quiet mode, just waiting..."),
                    ("Ω", "Oh! I'm observing something!"),
                    ("Ω θ", "Now I'm observing and thinking about it..."),
                    ("Ω! Ξ", "Wow! I see a pattern here!"),
                    ("μ", "Let me store this in my memory...")
                ]
                
                for state, description in states:
                    print(f"\nState: {state}")
                    cas.speak(description, mood='curious')
                    time.sleep(1.5)
                    
            elif user_input.lower() == '/test':
                print("\nTesting all voice moods...")
                test_phrases = [
                    ('neutral', "This is my normal voice."),
                    ('excited', "Oh wow! This is so exciting!"),
                    ('curious', "Hmm, I wonder what this means?"),
                    ('sleepy', "I'm feeling rather tired now...")
                ]
                
                for mood, phrase in test_phrases:
                    print(f"\nMood: {mood}")
                    cas.speak(phrase, mood=mood)
                    time.sleep(1.5)
                    
            elif user_input.startswith('/excited '):
                text = user_input[9:]
                cas.speak(text, mood='excited')
                
            elif user_input.startswith('/curious '):
                text = user_input[9:]
                cas.speak(text, mood='curious')
                
            elif user_input.startswith('/sleepy '):
                text = user_input[8:]
                cas.speak(text, mood='sleepy')
                
            else:
                # Default: speak with neutral mood
                cas.speak(user_input)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted!")
            cas.speak("Oh! You interrupted me!", mood='excited')
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThanks for testing consciousness audio in WSL!")


if __name__ == "__main__":
    main()