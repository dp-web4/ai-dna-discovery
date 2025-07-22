#!/usr/bin/env python3
"""
Demonstration of portable consciousness audio system
Works with real hardware or simulation mode
"""

import os
import sys
import time
import argparse
import queue
import threading
from typing import Optional
from datetime import datetime
import numpy as np

# Add audio-system to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consciousness_audio_system import ConsciousnessAudioSystem
from audio_hal_simulator import create_simulated_hal, SimulatedTTS


class DemoConsciousnessAudio(ConsciousnessAudioSystem):
    """Extended consciousness audio system with simulation support"""
    
    def __init__(self, simulate: bool = False, config_file: Optional[str] = None):
        self.simulate = simulate
        
        if simulate:
            print("🎭 Running in SIMULATION mode\n")
            # Don't call parent init, set up manually
            self.config_file = config_file
            self.is_running = False
            self.audio_queue = queue.Queue()
            self.event_queue = queue.Queue()
            
            # Consciousness state
            self.current_state = "..."
            self.state_history = []
            
            # Callbacks
            self.on_audio_event = None
            self.on_state_change = None
            
            # Use simulated HAL
            self.hal = create_simulated_hal(config_file)
            
            # TTS settings
            self.tts_engine = 'simulator'
            self._simulated_speak = SimulatedTTS.speak
            self.voice_settings = {'rate': 150, 'pitch': 50, 'voice': 'simulated'}
            
            # Audio settings from HAL
            self.settings = self.hal.audio_settings.copy()
        else:
            # Normal initialization
            super().__init__(config_file)
    
    def speak(self, text: str, mood: str = 'neutral'):
        """Speak with real or simulated TTS"""
        if self.simulate and hasattr(self, '_simulated_speak'):
            # Calculate mood adjustments
            rate = self.voice_settings['rate']
            pitch = self.voice_settings['pitch']
            
            if mood == 'excited':
                rate = int(rate * 1.2)
                pitch = int(pitch * 1.1)
            elif mood == 'curious':
                rate = int(rate * 0.9)
                pitch = int(pitch * 1.05)
            elif mood == 'sleepy':
                rate = int(rate * 0.7)
                pitch = int(pitch * 0.9)
            
            self._simulated_speak(text, voice='simulated', rate=rate, pitch=pitch)
        else:
            super().speak(text, mood)
    
    def demonstrate_full_system(self):
        """Full system demonstration"""
        print("\n=== Consciousness Audio System Demo ===")
        print(f"Mode: {'SIMULATION' if self.simulate else 'HARDWARE'}")
        print(f"Platform: {self.hal.platform['system']} ({self.hal.platform['device_type']})")
        print(f"Audio Backend: {self.hal.active_backend.__class__.__name__}")
        print(f"TTS Engine: {self.tts_engine}\n")
        
        # 1. Introduction
        self.speak("Welcome to the consciousness audio demonstration!", mood='excited')
        time.sleep(1)
        
        # 2. Test different moods
        print("Testing voice moods...")
        moods = [
            ('neutral', "This is my normal voice."),
            ('excited', "Wow! I'm so excited to be talking with you!"),
            ('curious', "Hmm, I wonder what that sound was?"),
            ('sleepy', "I'm feeling a bit sleepy now...")
        ]
        
        for mood, text in moods:
            print(f"\nMood: {mood}")
            self.speak(text, mood=mood)
            time.sleep(2)
        
        # 3. Consciousness states demo
        print("\n\nDemonstrating consciousness states...")
        self.demonstrate_consciousness()
        
        # 4. Audio monitoring demo
        print("\n\nStarting audio monitoring for 10 seconds...")
        self.speak("Now I'll listen for sounds and react to them!", mood='excited')
        
        # Set up reactions
        def react_to_state_change(old_state, new_state):
            reactions = {
                ("...", "Ω"): ("I hear something!", 'curious'),
                ("Ω", "Ω θ"): ("Let me think about that...", 'neutral'),
                ("Ω θ", "Ω! Ξ"): ("Oh! That's a pattern I recognize!", 'excited'),
                ("Ω! Ξ", "..."): ("It's quiet again.", 'sleepy')
            }
            
            key = (old_state, new_state)
            if key in reactions:
                text, mood = reactions[key]
                self.speak(text, mood=mood)
        
        self.on_state_change = react_to_state_change
        
        # Visual display
        def display_audio(event):
            level = event['level']
            state = event['state']
            
            # Create visual representation
            bar_length = int(level * 30 / self.settings.get('gain', 1.0))
            bar = '█' * bar_length + '░' * (30 - bar_length)
            
            # Add state visualization
            state_visual = {
                "...": "😴",
                "Ω": "👁️",
                "Ω θ": "👁️🤔",
                "Ω! Ξ": "👁️⚡🔍",
                "μ": "💾"
            }.get(state, "❓")
            
            print(f"\r[{bar}] {level:.3f} {state_visual} {state}    ", end='', flush=True)
        
        self.on_audio_event = display_audio
        
        # Start monitoring
        if self.start():
            time.sleep(10)
            self.stop()
        
        print("\n\n✅ Demo complete!")
        self.speak("That was fun! Thanks for watching the demo!", mood='excited')


def main():
    parser = argparse.ArgumentParser(description='Portable Consciousness Audio Demo')
    parser.add_argument('--simulate', '-s', action='store_true',
                       help='Run in simulation mode (no audio hardware needed)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run interactive mode instead of demo')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to audio configuration file')
    
    args = parser.parse_args()
    
    # Create system
    system = DemoConsciousnessAudio(simulate=args.simulate, config_file=args.config)
    
    try:
        if args.interactive:
            # Run interactive mode
            system.interactive_mode()
        else:
            # Run full demo
            system.demonstrate_full_system()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        system.speak("Goodbye!", mood='sleepy')
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if not args.simulate:
            print("\nTry running with --simulate flag for testing without audio hardware:")
            print("  python3 demo_portable_audio.py --simulate")


if __name__ == "__main__":
    main()