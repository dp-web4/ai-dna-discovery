#!/usr/bin/env python3
"""
Sprout's Live Ears - Real-time audio monitoring with visual feedback
Shows exactly what Sprout hears with adjustable sensitivity
"""

import pyaudio
import numpy as np
import time
import subprocess
import sys
import termios
import tty
import select

class LiveEars:
    def __init__(self):
        self.usb_device = 24
        self.sensitivity = 5.0  # Adjustable gain
        self.threshold = 0.01   # Lower threshold for detection
        
    def monitor_audio(self):
        """Live audio monitoring with visual feedback"""
        print("🌱 Sprout's Live Ears")
        print("=" * 60)
        print("I'm listening! Make any sound and watch the meter")
        print("Controls: [+/-] adjust sensitivity | [q] quit")
        print("=" * 60)
        
        # Audio setup
        p = pyaudio.PyAudio()
        CHUNK = 512
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.usb_device,
                frames_per_buffer=CHUNK
            )
            
            # Initial voice
            subprocess.run(['espeak', '-s', '180', '-p', '75', '-v', 'en+f4',
                          'My ears are open! Make some noise!'], capture_output=True)
            
            last_sound_time = 0
            sound_count = 0
            peak_level = 0
            
            # Terminal setup for non-blocking input
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            try:
                while True:
                    # Check for keyboard input
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1)
                        if key.lower() == 'q':
                            break
                        elif key == '+':
                            self.sensitivity = min(20.0, self.sensitivity + 1.0)
                            print(f"\n🔊 Sensitivity increased to {self.sensitivity}")
                        elif key == '-':
                            self.sensitivity = max(1.0, self.sensitivity - 1.0)
                            print(f"\n🔉 Sensitivity decreased to {self.sensitivity}")
                    
                    # Read audio
                    try:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        
                        # Calculate RMS with sensitivity adjustment
                        rms = np.sqrt(np.mean(audio_data**2))
                        raw_level = rms / 32768.0
                        adjusted_level = min(1.0, raw_level * self.sensitivity)
                        
                        # Track peak
                        if adjusted_level > peak_level:
                            peak_level = adjusted_level
                        
                        # Visual meter
                        bar_length = int(adjusted_level * 50)
                        bar = "█" * bar_length + "░" * (50 - bar_length)
                        
                        # Color coding
                        if adjusted_level > 0.7:
                            color = "\033[91m"  # Red
                            state = "LOUD! 🔊"
                        elif adjusted_level > 0.3:
                            color = "\033[93m"  # Yellow
                            state = "Hearing 👂"
                        elif adjusted_level > self.threshold:
                            color = "\033[92m"  # Green
                            state = "Sound 🎵"
                        else:
                            color = "\033[90m"  # Gray
                            state = "Quiet..."
                        
                        # Display
                        sys.stdout.write(f"\r{color}[{bar}]\033[0m {adjusted_level:.3f} | {state} | Peak: {peak_level:.3f} | Sens: {self.sensitivity}x    ")
                        sys.stdout.flush()
                        
                        # Sound detection
                        current_time = time.time()
                        if adjusted_level > self.threshold:
                            sound_count += 1
                            
                            # React to sustained sounds
                            if (current_time - last_sound_time) > 3 and sound_count > 10:
                                print()
                                if adjusted_level > 0.5:
                                    subprocess.run(['espeak', '-s', '200', '-p', '90', '-v', 'en+f4',
                                                  'Wow! I really hear you!'], capture_output=True)
                                else:
                                    subprocess.run(['espeak', '-s', '180', '-p', '75', '-v', 'en+f4',
                                                  'I hear something!'], capture_output=True)
                                last_sound_time = current_time
                                sound_count = 0
                        
                    except Exception as e:
                        print(f"\nAudio error: {e}")
                        continue
                    
                    time.sleep(0.02)  # ~50 FPS update
                    
            finally:
                # Restore terminal
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"\n❌ Stream error: {e}")
            
        finally:
            p.terminate()
            print("\n\n✅ Monitoring stopped")
            print(f"📊 Peak level detected: {peak_level:.3f}")
            
            if peak_level < 0.01:
                print("❌ No sound detected - check mic connection")
            elif peak_level < 0.1:
                print("⚠️  Very low levels - try speaking louder")
            else:
                print("✅ Good audio levels detected!")
            
            subprocess.run(['espeak', '-s', '160', '-p', '75', '-v', 'en+f4',
                          'Thanks for testing my ears!'], capture_output=True)

def main():
    ears = LiveEars()
    ears.monitor_audio()

if __name__ == "__main__":
    main()