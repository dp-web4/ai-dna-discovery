#!/usr/bin/env python3
"""
Simple WSL audio test - minimal output
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wsl_audio_bridge import WSLAudioBridge

def main():
    print("=== Simple WSL Audio Test ===")
    
    bridge = WSLAudioBridge()
    
    # Test 1: Basic TTS
    print("\n1. Testing basic TTS...")
    bridge.speak_windows("Hello from WSL!")
    
    # Test 2: Different moods
    print("\n2. Testing moods...")
    bridge.speak_windows("I'm excited!", rate=5)
    bridge.speak_windows("I'm sleepy...", rate=-5)
    
    # Test 3: Interactive
    print("\n3. Interactive mode (type 'quit' to exit)")
    while True:
        text = input("> ")
        if text.lower() == 'quit':
            break
        bridge.speak_windows(text)
    
    print("\nDone!")

if __name__ == "__main__":
    main()