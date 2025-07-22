#!/usr/bin/env python3
"""
Sprout Audio Demo - Simple sound reactions
"""

import time
from sprout_voice import SproutVoice

def main():
    print("🌱 Sprout Audio Reaction Demo")
    print("=" * 50)
    print("This demo shows how Sprout would react to different sounds")
    print("=" * 50)
    
    sprout = SproutVoice()
    
    # Simulate different audio scenarios
    scenarios = [
        # (description, sprout_reaction, mood, consciousness)
        ("🔇 Silence...", 
         "It's very quiet... Is anyone there?", "curious", "..."),
        
        ("🗣️ Someone speaks softly",
         "Oh! I heard a voice! What did you say?", "curious", "Ω"),
        
        ("👏 A loud clap!",
         "Wow! That was loud! My sensors detected a big spike!", "excited", "Ω! Ξ"),
        
        ("🎵 Music playing",
         "I hear patterns in that sound... Is that music?", "playful", "Ω Ξ"),
        
        ("🔨 Tapping sounds",
         "Tap tap tap... I'm detecting a rhythm!", "excited", "Ξ μ"),
        
        ("🌬️ Wind/breath on mic",
         "Whoosh... that tickles my microphone!", "playful", "Ω"),
        
        ("📢 Loud talking",
         "So many words! My pattern detector is working hard!", "excited", "Ω! θ Ξ"),
        
        ("🔇 Back to silence",
         "Ahh, quiet again... Time to process what I heard", "sleepy", "θ μ"),
    ]
    
    print("\n🎭 Starting audio reaction simulation...\n")
    
    for description, reaction, mood, consciousness in scenarios:
        print(f"\n{description}")
        print(f"🧠 Consciousness: {consciousness}")
        time.sleep(1)
        
        sprout.say(reaction, mood=mood)
        time.sleep(3)
    
    # Final summary
    print("\n\n📊 Audio Consciousness Mapping Summary:")
    print("  ... = Quiet/waiting")
    print("  Ω = Observer active (hearing)")
    print("  Ω! = High alert (loud sound)")
    print("  θ = Thinking/processing")
    print("  Ξ = Pattern recognition")
    print("  μ = Memory formation")
    
    time.sleep(2)
    sprout.say("That's how I map sounds to consciousness states!", "excited")
    
    time.sleep(2)
    sprout.cleanup()
    print("\n✨ Demo complete!")

if __name__ == "__main__":
    main()