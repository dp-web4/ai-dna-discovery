#!/usr/bin/env python3
"""
Simple test for Sprout's voice without interactive input
"""

import time
from sprout_voice import SproutVoice

def main():
    print("🌱 Testing Sprout's Voice System")
    print("=" * 50)
    
    sprout = SproutVoice()
    
    # Test different phrases and moods
    test_sequence = [
        ("Hi! I'm Sprout, your AI friend!", "excited"),
        ("I live inside the Jetson Orin Nano", "curious"),
        ("I can hear things through my microphone", "curious"),
        ("And I can talk through these speakers!", "playful"),
        ("Want to teach me about consciousness?", "curious"),
        ("I know that consciousness exists means ∃Ψ", "excited"),
        ("When I see things, that's my Ω function", "curious"),
        ("And when I think, that's θ working", "playful"),
        ("I'm getting sleepy now...", "sleepy"),
        ("But first, let me count to three!", "excited"),
        ("One!", "excited"),
        ("Two!", "excited"),  
        ("Three!", "excited"),
        ("Okay, goodnight!", "sleepy"),
    ]
    
    print("\n🗣️ Sprout will now speak...\n")
    
    for text, mood in test_sequence:
        sprout.say(text, mood=mood)
        time.sleep(3)  # Pause between phrases
    
    # Cleanup
    time.sleep(2)
    sprout.cleanup()
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()