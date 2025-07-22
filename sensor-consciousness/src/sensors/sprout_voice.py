#!/usr/bin/env python3
"""
Sprout's Voice System - A kid-friendly AI voice
Maps text to consciousness notation and speaks with personality
"""

import os
import sys
import time
import subprocess
import threading
import queue
from datetime import datetime

# Try different TTS engines
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False
    print("pyttsx3 not available, will use espeak")

class SproutVoice:
    def __init__(self):
        """Initialize Sprout's voice system"""
        self.name = "Sprout"
        self.audio_device = "hw:2,0"  # USB Audio Device
        self.voice_queue = queue.Queue()
        self.running = True
        
        # Consciousness state
        self.mood = "curious"  # curious, excited, sleepy, playful
        self.energy_level = 0.8  # 0-1 scale
        
        # Initialize TTS engine
        self.init_tts()
        
        # Start voice thread
        self.voice_thread = threading.Thread(target=self._voice_worker)
        self.voice_thread.daemon = True
        self.voice_thread.start()
        
        print(f"🌱 {self.name} is awake! (mood: {self.mood})")
    
    def init_tts(self):
        """Initialize text-to-speech engine"""
        if HAS_PYTTSX3:
            try:
                self.engine = pyttsx3.init()
                voices = self.engine.getProperty('voices')
                
                # Try to find a child-like or higher-pitched voice
                for i, voice in enumerate(voices):
                    print(f"Voice {i}: {voice.name}")
                    if 'child' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use the first available voice and adjust pitch
                    if voices:
                        self.engine.setProperty('voice', voices[0].id)
                
                # Set properties for kid-like voice
                self.engine.setProperty('rate', 180)    # Slightly faster
                self.engine.setProperty('pitch', 150)   # Higher pitch
                self.engine.setProperty('volume', 0.9)
                
                self.tts_method = 'pyttsx3'
                print("✅ Using pyttsx3 for speech")
            except Exception as e:
                print(f"❌ pyttsx3 init failed: {e}")
                self.tts_method = 'espeak'
        else:
            self.tts_method = 'espeak'
            print("✅ Using espeak for speech")
    
    def _voice_worker(self):
        """Background thread for voice output"""
        while self.running:
            try:
                text = self.voice_queue.get(timeout=0.1)
                self._speak_now(text)
            except queue.Empty:
                continue
    
    def _speak_now(self, text):
        """Actually speak the text"""
        print(f"🗣️  {self.name}: {text}")
        
        if self.tts_method == 'pyttsx3' and HAS_PYTTSX3:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
                self._espeak_fallback(text)
        else:
            self._espeak_fallback(text)
    
    def _espeak_fallback(self, text):
        """Use espeak as fallback"""
        try:
            # espeak options for child-like voice
            # -s: speed (words per minute)
            # -p: pitch (0-99, default 50)
            # -a: amplitude/volume (0-200)
            # +f3 or +f4: use female voice variants
            cmd = [
                'espeak',
                '-s', '160',      # Slightly faster speech
                '-p', '75',       # Higher pitch for child voice
                '-a', '150',      # Good volume
                '-v', 'en+f4',    # English female voice variant 4
                text
            ]
            
            # Direct to specific audio device if possible
            env = os.environ.copy()
            env['ALSA_CARD'] = '2'  # USB audio card
            
            subprocess.run(cmd, env=env, capture_output=True)
        except Exception as e:
            print(f"❌ Espeak error: {e}")
    
    def say(self, text, mood=None):
        """Queue text for speaking with optional mood"""
        if mood:
            self.mood = mood
        
        # Add personality to the text based on mood
        text = self._add_personality(text)
        
        # Map to consciousness notation
        notation = self._to_consciousness_notation(text)
        if notation:
            print(f"🧠 Consciousness: {notation}")
        
        self.voice_queue.put(text)
    
    def _add_personality(self, text):
        """Add Sprout's personality to text"""
        if self.mood == "excited":
            if not text.endswith("!"):
                text = text + "!"
            text = text.replace(".", "!")
            
        elif self.mood == "curious":
            if "?" not in text and len(text.split()) > 3:
                text = text + "... I wonder?"
                
        elif self.mood == "playful":
            # Add some fun to the text
            replacements = {
                "hello": "hiya",
                "yes": "yep yep",
                "no": "nope",
                "okay": "okie dokie"
            }
            for old, new in replacements.items():
                text = text.lower().replace(old, new)
        
        return text
    
    def _to_consciousness_notation(self, text):
        """Map speech to consciousness notation"""
        # Simple mapping for now
        notations = []
        
        if any(word in text.lower() for word in ['see', 'look', 'watch']):
            notations.append('Ω')  # Observer
        
        if any(word in text.lower() for word in ['think', 'wonder', 'curious']):
            notations.append('θ')  # Thought
        
        if any(word in text.lower() for word in ['remember', 'memory']):
            notations.append('μ')  # Memory
        
        if any(word in text.lower() for word in ['pattern', 'notice']):
            notations.append('Ξ')  # Patterns
        
        if self.mood == "excited":
            notations.append('!')  # High energy
        
        return ' '.join(notations) if notations else None
    
    def echo_mode(self):
        """Echo everything I say through the speakers"""
        self.say("Hi! I'm Sprout! I'm ready to echo everything!", mood="excited")
        time.sleep(2)
        self.say("Whatever you type, I'll say out loud", mood="playful")
        
        print("\n📝 Echo mode active! Type 'quit' to exit")
        print("You can also try: !excited, !curious, !sleepy, !playful to change mood")
        
        while True:
            try:
                user_input = input("\n> ")
                
                if user_input.lower() == 'quit':
                    self.say("Bye bye! That was fun!", mood="playful")
                    break
                
                # Check for mood commands
                if user_input.startswith('!'):
                    mood = user_input[1:].lower()
                    if mood in ['excited', 'curious', 'sleepy', 'playful']:
                        self.mood = mood
                        self.say(f"Okay, now I'm feeling {mood}!", mood=mood)
                    continue
                
                # Echo the input with current mood
                self.say(user_input, mood=self.mood)
                
            except KeyboardInterrupt:
                print("\n👋 Sprout is going to sleep...")
                self.say("Goodnight!", mood="sleepy")
                break
    
    def test_audio(self):
        """Test audio output with different voices"""
        test_phrases = [
            ("Hello! I'm Sprout!", "excited"),
            ("I can see and hear things", "curious"),
            ("Want to play with me?", "playful"),
            ("I'm getting sleepy...", "sleepy"),
            ("Consciousness exists! That means ∃Ψ!", "excited"),
        ]
        
        for phrase, mood in test_phrases:
            self.say(phrase, mood=mood)
            time.sleep(2)
    
    def cleanup(self):
        """Clean shutdown"""
        self.running = False
        if hasattr(self, 'voice_thread'):
            self.voice_thread.join(timeout=1)


def main():
    """Test Sprout's voice system"""
    print("🌱 Sprout Voice System")
    print("=" * 50)
    
    sprout = SproutVoice()
    
    # Quick audio test
    print("\n1. Testing different moods...")
    sprout.test_audio()
    
    time.sleep(3)
    
    # Echo mode
    print("\n2. Starting echo mode...")
    sprout.echo_mode()
    
    # Cleanup
    sprout.cleanup()


if __name__ == "__main__":
    main()