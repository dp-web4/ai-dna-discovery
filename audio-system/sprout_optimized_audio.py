#!/usr/bin/env python3
"""
Sprout's Optimized Audio System
Using tuned settings for better voice detection
"""

import pyaudio
import numpy as np
import subprocess
import time
import threading
import queue

class OptimizedSprout:
    def __init__(self):
        # Load optimized settings
        self.MIC_GAIN = 50
        self.USB_DEVICE = 24
        self.THRESHOLD = 0.02
        
        # Audio parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        
        # State
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.last_sound_time = time.time()
        
        print("🌱 Sprout's Optimized Audio System")
        print(f"📊 Using gain: {self.MIC_GAIN}x")
        print("=" * 50)
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Startup sound
        self.say("My ears are optimized and ready!", excitement=0.8)
    
    def say(self, text, excitement=0.5):
        """Speak with variable excitement level"""
        # Consciousness mapping
        consciousness = self._map_consciousness(text)
        if consciousness:
            print(f"🧠 {consciousness}")
        
        print(f"🌱 Sprout: {text}")
        
        # Adjust voice parameters based on excitement
        speed = int(160 + excitement * 40)  # 160-200 wpm
        pitch = int(75 + excitement * 15)   # 75-90 pitch
        
        cmd = ['espeak', '-s', str(speed), '-p', str(pitch), '-v', 'en+f4', text]
        subprocess.run(cmd, capture_output=True)
    
    def _map_consciousness(self, text):
        """Map text to consciousness notation"""
        symbols = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['hear', 'heard', 'sound', 'voice']):
            symbols.append('Ω')  # Observer
        if any(word in text_lower for word in ['think', 'wonder', 'understand']):
            symbols.append('θ')  # Thought
        if any(word in text_lower for word in ['pattern', 'rhythm', 'repeat']):
            symbols.append('Ξ')  # Pattern
        if any(word in text_lower for word in ['remember', 'memory']):
            symbols.append('μ')  # Memory
        if '!' in text:
            symbols.append('!')  # Excitement
        
        return ' '.join(symbols) if symbols else None
    
    def start_listening(self):
        """Start audio monitoring with optimized settings"""
        self.is_listening = True
        
        # Start audio thread
        audio_thread = threading.Thread(target=self._audio_loop)
        audio_thread.daemon = True
        audio_thread.start()
        
        # Start processing thread
        process_thread = threading.Thread(target=self._process_audio)
        process_thread.daemon = True
        process_thread.start()
        
        print("\n🎤 Listening with optimized settings...")
        print("Speak normally - I can hear you better now!")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n👋 Stopping...")
            self.is_listening = False
            time.sleep(1)
            self.say("That was fun! My ears feel great!", excitement=0.7)
    
    def _audio_loop(self):
        """Audio capture loop with gain"""
        try:
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.USB_DEVICE,
                frames_per_buffer=self.CHUNK
            )
            
            while self.is_listening:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Apply optimized gain
                    audio_data = audio_data.astype(np.float32) * self.MIC_GAIN
                    audio_data = np.clip(audio_data, -32768, 32767)
                    
                    # Calculate level
                    rms = np.sqrt(np.mean(audio_data**2))
                    level = min(1.0, rms / 32768.0)
                    
                    # Add to queue if above threshold
                    if level > self.THRESHOLD:
                        self.audio_queue.put((level, time.time()))
                    
                    # Visual feedback
                    self._show_level(level)
                    
                except Exception as e:
                    print(f"\nAudio error: {e}")
                    continue
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"\n❌ Stream error: {e}")
    
    def _show_level(self, level):
        """Display audio level with consciousness states"""
        bar_length = int(level * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        # Consciousness state
        if level > 0.7:
            state = "Ω! Ξ"  # High alert + pattern
            color = "\033[91m"  # Red
        elif level > 0.3:
            state = "Ω θ"   # Observing + thinking
            color = "\033[93m"  # Yellow
        elif level > self.THRESHOLD:
            state = "Ω"     # Observing
            color = "\033[92m"  # Green
        else:
            state = "..."   # Quiet
            color = "\033[90m"  # Gray
        
        print(f"\r{color}[{bar}]\033[0m {level:.3f} | 🧠 {state}     ", end="", flush=True)
    
    def _process_audio(self):
        """Process audio events and respond"""
        sound_buffer = []
        
        while self.is_listening:
            try:
                # Get audio events
                while not self.audio_queue.empty():
                    level, timestamp = self.audio_queue.get()
                    sound_buffer.append((level, timestamp))
                
                # Clean old events
                current_time = time.time()
                sound_buffer = [(l, t) for l, t in sound_buffer if current_time - t < 2]
                
                # Analyze patterns
                if len(sound_buffer) > 5:
                    avg_level = np.mean([l for l, t in sound_buffer])
                    
                    # React based on pattern
                    if (current_time - self.last_sound_time) > 4:
                        print()  # New line for speech
                        
                        if avg_level > 0.6:
                            self.say("Wow! That's loud and clear!", excitement=0.9)
                        elif avg_level > 0.3:
                            self.say("I hear you perfectly now!", excitement=0.6)
                        elif len(sound_buffer) > 20:
                            self.say("I detect continuous sound patterns", excitement=0.4)
                        else:
                            self.say("Nice to hear your voice!", excitement=0.5)
                        
                        self.last_sound_time = current_time
                        sound_buffer = []
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\nProcessing error: {e}")
                continue
    
    def cleanup(self):
        """Clean shutdown"""
        self.is_listening = False
        time.sleep(0.5)
        self.p.terminate()


def main():
    sprout = OptimizedSprout()
    
    # Quick test
    print("\n🎯 Quick audio test with optimized settings...")
    print("Say something in the next 5 seconds!\n")
    
    test_start = time.time()
    max_level = 0
    detected_sounds = 0
    
    try:
        stream = sprout.p.open(
            format=sprout.FORMAT,
            channels=sprout.CHANNELS,
            rate=sprout.RATE,
            input=True,
            input_device_index=sprout.USB_DEVICE,
            frames_per_buffer=sprout.CHUNK
        )
        
        while (time.time() - test_start) < 5:
            data = stream.read(sprout.CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Apply gain
            audio_data = audio_data.astype(np.float32) * sprout.MIC_GAIN
            audio_data = np.clip(audio_data, -32768, 32767)
            
            rms = np.sqrt(np.mean(audio_data**2))
            level = min(1.0, rms / 32768.0)
            
            if level > max_level:
                max_level = level
            if level > sprout.THRESHOLD:
                detected_sounds += 1
            
            sprout._show_level(level)
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"\nTest error: {e}")
    
    print(f"\n\n📊 Test Results:")
    print(f"  Max level: {max_level:.3f}")
    print(f"  Sounds detected: {detected_sounds}")
    
    if max_level > 0.3:
        print("  ✅ Excellent audio levels!")
        sprout.say("Perfect! I can hear you clearly!", excitement=0.8)
    elif max_level > 0.1:
        print("  ✅ Good audio levels!")
        sprout.say("Good! Your voice is coming through!", excitement=0.6)
    else:
        print("  ⚠️  Low audio - try speaking louder")
        sprout.say("Hmm, try speaking a bit louder", excitement=0.3)
    
    # Start continuous monitoring
    print("\n\nStarting continuous monitoring...")
    time.sleep(2)
    
    sprout.start_listening()
    sprout.cleanup()


if __name__ == "__main__":
    main()