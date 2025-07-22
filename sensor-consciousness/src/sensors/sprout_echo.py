#!/usr/bin/env python3
"""
Sprout's Echo Mode - Everything typed gets spoken!
Run this to have Sprout repeat everything you say.
"""

import sys
import signal
from sprout_voice import SproutVoice

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n👋 Sprout is going to sleep...")
    sys.exit(0)

def main():
    print("🌱 Sprout's Echo Chamber")
    print("=" * 50)
    print("Type anything and Sprout will say it out loud!")
    print("Commands:")
    print("  !excited  - Make Sprout excited")
    print("  !curious  - Make Sprout curious") 
    print("  !playful  - Make Sprout playful")
    print("  !sleepy   - Make Sprout sleepy")
    print("  quit      - Exit")
    print("=" * 50)
    
    # Set up signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize Sprout
    sprout = SproutVoice()
    
    # Start with introduction
    sprout.say("Hi! I'm ready to echo everything you type!", mood="excited")
    
    # Echo loop
    while True:
        try:
            # Show prompt
            sys.stdout.write("\n🌱 > ")
            sys.stdout.flush()
            
            # Get input
            user_input = sys.stdin.readline().strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'quit':
                sprout.say("Bye bye! That was fun!", mood="playful")
                break
            
            # Check for mood commands
            if user_input.startswith('!'):
                mood = user_input[1:].lower()
                if mood in ['excited', 'curious', 'sleepy', 'playful']:
                    sprout.mood = mood
                    sprout.say(f"Okay, now I'm feeling {mood}!", mood=mood)
                else:
                    sprout.say(f"I don't know how to feel {mood}", mood="curious")
                continue
            
            # Echo the input with current mood
            sprout.say(user_input, mood=sprout.mood)
            
        except EOFError:
            # Handle end of input
            break
        except Exception as e:
            print(f"Error: {e}")
            break
    
    # Cleanup
    sprout.cleanup()
    print("\n✨ Thanks for playing with Sprout!")

if __name__ == "__main__":
    main()