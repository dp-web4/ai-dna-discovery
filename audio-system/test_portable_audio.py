#!/usr/bin/env python3
"""
Test portable audio system across different platforms
"""

import sys
import os

# Add audio-system to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_hal import AudioHAL
from consciousness_audio_system import ConsciousnessAudioSystem


def quick_test():
    """Quick test of audio capabilities"""
    print("=== Portable Audio System Test ===\n")
    
    # Test HAL
    print("1. Testing Audio HAL...")
    hal = AudioHAL()
    
    if not hal.backends:
        print("❌ No audio backends available!")
        print("\nTrying to install required packages...")
        
        # Suggest installation based on platform
        if hal.platform['system'] == 'Linux':
            print("On Linux/Jetson, try:")
            print("  sudo apt-get install python3-pyaudio")
            print("  pip3 install sounddevice")
        elif hal.platform['system'] == 'Darwin':
            print("On macOS, try:")
            print("  brew install portaudio")
            print("  pip3 install pyaudio sounddevice")
        elif hal.platform['system'] == 'Windows':
            print("On Windows, try:")
            print("  pip3 install pyaudio sounddevice")
        
        return False
    
    print(f"✅ Found {len(hal.backends)} audio backend(s)")
    print(f"✅ Active backend: {hal.active_backend.__class__.__name__ if hal.active_backend else 'None'}")
    
    if hal.input_device:
        print(f"✅ Input device: {hal.input_device.name}")
    else:
        print("❌ No input device found")
    
    if hal.output_device:
        print(f"✅ Output device: {hal.output_device.name}")
    else:
        print("❌ No output device found")
    
    # Test consciousness system
    print("\n2. Testing Consciousness Audio System...")
    cas = ConsciousnessAudioSystem()
    
    print(f"✅ TTS Engine: {cas.tts_engine}")
    
    if cas.tts_engine != 'none':
        print("\n3. Testing Text-to-Speech...")
        cas.speak("Audio system initialized successfully!")
    else:
        print("❌ No TTS engine available")
    
    # Show how to run interactive mode
    print("\n✅ System appears ready!")
    print("\nTo run interactive mode:")
    print("  python3 consciousness_audio_system.py")
    
    return True


def install_dependencies():
    """Helper to install dependencies"""
    print("\n=== Installing Audio Dependencies ===")
    
    import subprocess
    
    # Detect platform
    if sys.platform.startswith('linux'):
        print("Installing Linux dependencies...")
        cmds = [
            ['sudo', 'apt-get', 'update'],
            ['sudo', 'apt-get', 'install', '-y', 'python3-pyaudio', 'portaudio19-dev', 'espeak'],
            ['pip3', 'install', 'sounddevice', 'numpy']
        ]
    elif sys.platform == 'darwin':
        print("Installing macOS dependencies...")
        cmds = [
            ['brew', 'install', 'portaudio'],
            ['pip3', 'install', 'pyaudio', 'sounddevice', 'numpy']
        ]
    else:
        print("Installing Windows dependencies...")
        cmds = [
            ['pip3', 'install', 'pyaudio', 'sounddevice', 'numpy', 'pyttsx3']
        ]
    
    for cmd in cmds:
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("✅ Success")
        except subprocess.CalledProcessError:
            print("❌ Failed (may need manual installation)")
        except FileNotFoundError:
            print("❌ Command not found")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test portable audio system')
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    args = parser.parse_args()
    
    if args.install:
        install_dependencies()
    else:
        success = quick_test()
        
        if not success:
            print("\nRun with --install flag to install dependencies:")
            print("  python3 test_portable_audio.py --install")