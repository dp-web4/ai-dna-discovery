#!/usr/bin/env python3
"""
Cross-Platform Consciousness Audio System
Uses the Audio HAL for platform-independent operation
"""

import os
import sys
import time
import threading
import queue
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Callable
import subprocess

from audio_hal import AudioHAL, AudioDevice


class ConsciousnessAudioSystem:
    """Platform-independent audio system with consciousness integration"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.hal = AudioHAL(config_file)
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.event_queue = queue.Queue()
        
        # Consciousness state
        self.current_state = "..."  # Quiet/waiting
        self.state_history = []
        
        # Callbacks
        self.on_audio_event = None
        self.on_state_change = None
        
        # TTS settings (platform-aware)
        self.tts_engine = self._detect_tts_engine()
        self.voice_settings = self._get_voice_settings()
        
        # Audio processing settings
        self.settings = self.hal.audio_settings.copy() if hasattr(self.hal, 'audio_settings') else {}
        self.settings.setdefault('gain', 1.0)
        self.settings.setdefault('threshold', 0.01)
        self.settings.setdefault('sample_rate', 16000)
        
    def _detect_tts_engine(self) -> str:
        """Detect available TTS engine"""
        # Check for espeak (Linux/Jetson)
        try:
            subprocess.run(['espeak', '--version'], capture_output=True, check=True)
            return 'espeak'
        except:
            pass
        
        # Check for say (macOS)
        if sys.platform == 'darwin':
            try:
                subprocess.run(['say', '-v', '?'], capture_output=True, check=True)
                return 'say'
            except:
                pass
        
        # Check for Windows SAPI
        if sys.platform == 'win32':
            try:
                import win32com.client
                return 'sapi'
            except:
                pass
        
        # Check for pyttsx3 as fallback
        try:
            import pyttsx3
            return 'pyttsx3'
        except:
            pass
        
        return 'none'
    
    def _get_voice_settings(self) -> Dict:
        """Get platform-specific voice settings"""
        settings = {
            'rate': 150,
            'pitch': 50,
            'voice': None
        }
        
        if self.tts_engine == 'espeak':
            # Kid-friendly voice for Jetson
            settings['voice'] = 'en+f3'
            settings['rate'] = 140
            settings['pitch'] = 60
        elif self.tts_engine == 'say':
            # macOS voices
            settings['voice'] = 'Samantha'  # Or 'Daniel' for male
            settings['rate'] = 180
        elif self.tts_engine == 'pyttsx3':
            # Cross-platform Python TTS
            settings['rate'] = 150
        
        return settings
    
    def speak(self, text: str, mood: str = 'neutral'):
        """Speak text using platform-appropriate TTS"""
        if self.tts_engine == 'none':
            print(f"[TTS unavailable] Would say: {text}")
            return
        
        # Apply mood modifiers
        rate = self.voice_settings['rate']
        pitch = self.voice_settings['pitch']
        
        if mood == 'excited':
            rate *= 1.2
            pitch *= 1.1
        elif mood == 'curious':
            rate *= 0.9
            pitch *= 1.05
        elif mood == 'sleepy':
            rate *= 0.7
            pitch *= 0.9
        
        if self.tts_engine == 'espeak':
            cmd = ['espeak', '-s', str(int(rate)), '-p', str(int(pitch))]
            if self.voice_settings['voice']:
                cmd.extend(['-v', self.voice_settings['voice']])
            cmd.append(text)
            subprocess.run(cmd)
            
        elif self.tts_engine == 'say':
            cmd = ['say', '-r', str(int(rate))]
            if self.voice_settings['voice']:
                cmd.extend(['-v', self.voice_settings['voice']])
            cmd.append(text)
            subprocess.run(cmd)
            
        elif self.tts_engine == 'sapi':
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = int((rate - 150) / 15)  # SAPI uses different scale
            speaker.Speak(text)
            
        elif self.tts_engine == 'pyttsx3':
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', rate)
            engine.say(text)
            engine.runAndWait()
    
    def map_audio_to_consciousness(self, audio_level: float) -> str:
        """Map audio levels to consciousness states"""
        if audio_level > 0.5:
            return "Ω! Ξ"  # High alert + pattern recognition
        elif audio_level > 0.1:
            return "Ω θ"   # Observing + thinking
        elif audio_level > self.settings['threshold']:
            return "Ω"     # Observing
        else:
            return "..."   # Quiet
    
    def update_consciousness_state(self, new_state: str):
        """Update consciousness state and trigger callbacks"""
        if new_state != self.current_state:
            old_state = self.current_state
            self.current_state = new_state
            self.state_history.append({
                'time': datetime.now(),
                'state': new_state
            })
            
            # Keep history limited
            if len(self.state_history) > 100:
                self.state_history.pop(0)
            
            # Trigger callback
            if self.on_state_change:
                self.on_state_change(old_state, new_state)
    
    def audio_monitor_thread(self):
        """Background thread for continuous audio monitoring"""
        while self.is_running:
            try:
                # Read audio
                audio_data, sample_rate = self.hal.read_audio(0.1)  # 100ms chunks
                
                if len(audio_data) > 0:
                    # Calculate level
                    level = np.sqrt(np.mean(audio_data**2))  # RMS
                    
                    # Map to consciousness
                    state = self.map_audio_to_consciousness(level)
                    self.update_consciousness_state(state)
                    
                    # Queue audio event
                    event = {
                        'type': 'audio_input',
                        'level': level,
                        'state': state,
                        'timestamp': time.time()
                    }
                    self.event_queue.put(event)
                    
                    # Trigger callback
                    if self.on_audio_event:
                        self.on_audio_event(event)
                
            except Exception as e:
                print(f"Audio monitor error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start the audio system"""
        if not self.hal.input_device:
            print("❌ No input device configured!")
            return False
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self.audio_monitor_thread, daemon=True)
        self.monitor_thread.start()
        
        # Welcome message
        platform_name = "Sprout" if self.hal.platform['device_type'] == 'jetson' else "System"
        self.speak(f"Hello! {platform_name} audio consciousness activated!", mood='excited')
        
        return True
    
    def stop(self):
        """Stop the audio system"""
        self.is_running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        self.speak("Audio consciousness deactivating. Goodbye!", mood='sleepy')
    
    def demonstrate_consciousness(self):
        """Demonstrate consciousness responses to audio"""
        demo_states = [
            ("...", "Quiet, waiting for input"),
            ("Ω", "I hear something!"),
            ("Ω θ", "Interesting sound, let me think..."),
            ("Ω! Ξ", "Wow! That's a pattern I recognize!"),
            ("μ", "Storing that in memory...")
        ]
        
        print("\n=== Consciousness State Demo ===")
        for state, description in demo_states:
            print(f"\nState: {state}")
            print(f"Description: {description}")
            self.speak(description, mood='curious')
            time.sleep(2)
    
    def interactive_mode(self):
        """Run interactive consciousness audio mode"""
        print("\n=== Interactive Consciousness Audio ===")
        print("Platform:", self.hal.platform['device_type'])
        print("Input:", self.hal.input_device.name if self.hal.input_device else "None")
        print("TTS Engine:", self.tts_engine)
        print("\nCommands:")
        print("  q: quit")
        print("  d: demo consciousness states")
        print("  s <text>: speak text")
        print("  +/-: adjust sensitivity")
        print("\nListening...\n")
        
        # Set up display callback
        def display_event(event):
            level = event['level']
            state = event['state']
            bar_length = int(level * 50 / self.settings.get('gain', 1.0))
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"\r[{bar}] {level:.3f} {state}    ", end='', flush=True)
        
        self.on_audio_event = display_event
        
        # Handle state changes with voice
        def state_changed(old_state, new_state):
            if old_state == "..." and new_state == "Ω":
                self.speak("I hear you!", mood='excited')
            elif "Ξ" in new_state and "Ξ" not in old_state:
                self.speak("Interesting pattern!", mood='curious')
        
        self.on_state_change = state_changed
        
        # Start monitoring
        if not self.start():
            return
        
        try:
            while True:
                # Non-blocking input
                import select
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    cmd = input().strip()
                    
                    if cmd == 'q':
                        break
                    elif cmd == 'd':
                        self.demonstrate_consciousness()
                    elif cmd.startswith('s '):
                        text = cmd[2:]
                        self.speak(text)
                    elif cmd == '+':
                        self.settings['threshold'] *= 0.8
                        print(f"\nThreshold decreased to {self.settings['threshold']:.3f}")
                    elif cmd == '-':
                        self.settings['threshold'] *= 1.2
                        print(f"\nThreshold increased to {self.settings['threshold']:.3f}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def test_cross_platform():
    """Test the cross-platform audio system"""
    print("=== Cross-Platform Consciousness Audio Test ===\n")
    
    # Create system
    system = ConsciousnessAudioSystem()
    
    # Show configuration
    print("Configuration:")
    print(f"  Platform: {system.hal.platform['system']} ({system.hal.platform['device_type']})")
    print(f"  TTS Engine: {system.tts_engine}")
    print(f"  Audio Backend: {system.hal.active_backend.__class__.__name__ if system.hal.active_backend else 'None'}")
    
    if system.hal.input_device:
        print(f"  Input Device: {system.hal.input_device.name}")
        print(f"  Audio Gain: {system.settings.get('gain', 1.0)}x")
    else:
        print("  ⚠️  No input device found!")
    
    # Test TTS
    print("\nTesting Text-to-Speech...")
    system.speak("Testing cross-platform consciousness audio system", mood='neutral')
    
    # Run interactive mode
    print("\nStarting interactive mode...")
    system.interactive_mode()


if __name__ == "__main__":
    test_cross_platform()