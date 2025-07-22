#!/usr/bin/env python3
"""
WSL Audio Bridge - Enables audio in Windows Subsystem for Linux
Uses Windows native capabilities through PowerShell
"""

import subprocess
import os
import tempfile
import wave
import numpy as np
from typing import Optional


class WSLAudioBridge:
    """Bridge WSL to Windows audio capabilities"""
    
    def __init__(self):
        self.powershell = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        self.has_powershell = os.path.exists(self.powershell)
        
    def speak_windows(self, text: str, rate: int = 0, voice: Optional[str] = None):
        """Use Windows SAPI through PowerShell"""
        if not self.has_powershell:
            print(f"[WSL] No PowerShell access: {text}")
            return
            
        # Escape quotes in text
        text = text.replace('"', '`"').replace("'", "`'")
        
        # Build PowerShell command
        ps_command = f'Add-Type -AssemblyName System.Speech; '
        ps_command += f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
        
        if voice:
            ps_command += f'$speak.SelectVoice("{voice}"); '
        
        if rate != 0:
            ps_command += f'$speak.Rate = {rate}; '
            
        ps_command += f'$speak.Speak("{text}")'
        
        # Execute through PowerShell
        try:
            subprocess.run([self.powershell, "-Command", ps_command], 
                         capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[WSL] TTS Error: {e}")
    
    def list_windows_voices(self):
        """List available Windows voices"""
        if not self.has_powershell:
            return []
            
        ps_command = '''
        Add-Type -AssemblyName System.Speech
        $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $speak.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
        '''
        
        try:
            result = subprocess.run([self.powershell, "-Command", ps_command],
                                  capture_output=True, text=True, check=True)
            voices = result.stdout.strip().split('\n')
            return [v.strip() for v in voices if v.strip()]
        except:
            return []
    
    def record_windows(self, duration: float, output_file: str):
        """Record audio using Windows sound recorder"""
        if not self.has_powershell:
            print("[WSL] No recording capability in WSL")
            return None
            
        # Convert WSL path to Windows path
        win_path = subprocess.run(['wslpath', '-w', output_file], 
                                capture_output=True, text=True).stdout.strip()
        
        # PowerShell script to record audio
        ps_command = f'''
        Add-Type -TypeDefinition @"
        using System;
        using System.IO;
        using System.Threading;
        using NAudio.Wave;
        
        public class AudioRecorder {{
            private WaveInEvent waveIn;
            private WaveFileWriter writer;
            
            public void Record(string filename, int seconds) {{
                waveIn = new WaveInEvent();
                waveIn.WaveFormat = new WaveFormat(16000, 1);
                
                writer = new WaveFileWriter(filename, waveIn.WaveFormat);
                waveIn.DataAvailable += (s, e) => writer.Write(e.Buffer, 0, e.BytesRecorded);
                
                waveIn.StartRecording();
                Thread.Sleep(seconds * 1000);
                waveIn.StopRecording();
                
                writer?.Dispose();
                waveIn?.Dispose();
            }}
        }}
        "@
        
        [AudioRecorder]::new().Record("{win_path}", {int(duration)})
        '''
        
        # Note: This requires NAudio on Windows side
        # For now, we'll indicate recording isn't available
        print(f"[WSL] Audio recording not available in WSL")
        return None
    
    def play_sound_windows(self, audio_file: str):
        """Play sound file through Windows"""
        if not self.has_powershell:
            return
            
        # Convert path if needed
        if audio_file.startswith('/'):
            win_path = subprocess.run(['wslpath', '-w', audio_file],
                                    capture_output=True, text=True).stdout.strip()
        else:
            win_path = audio_file
            
        ps_command = f'(New-Object Media.SoundPlayer "{win_path}").PlaySync()'
        
        try:
            subprocess.run([self.powershell, "-Command", ps_command],
                         capture_output=True, check=True)
        except:
            print(f"[WSL] Could not play {audio_file}")


# Integrate with our audio HAL
def create_wsl_audio_device():
    """Create a pseudo audio device for WSL"""
    from audio_hal import AudioDevice
    
    # Create virtual devices
    input_device = AudioDevice(
        index=0,
        name="WSL Virtual Microphone (No Recording)",
        channels_in=0,  # No actual input in WSL
        channels_out=0,
        sample_rate=16000,
        is_default=True
    )
    
    output_device = AudioDevice(
        index=1,
        name="Windows TTS via PowerShell",
        channels_in=0,
        channels_out=2,
        sample_rate=16000,
        is_default=True
    )
    
    return input_device, output_device


def test_wsl_audio():
    """Test WSL audio bridge"""
    print("=== WSL Audio Bridge Test ===\n")
    
    bridge = WSLAudioBridge()
    
    if bridge.has_powershell:
        print("✅ PowerShell access available")
        
        # List voices
        print("\nAvailable Windows voices:")
        voices = bridge.list_windows_voices()
        for i, voice in enumerate(voices):
            print(f"  {i+1}. {voice}")
        
        # Test TTS
        print("\nTesting Windows TTS...")
        bridge.speak_windows("Hello from Windows Subsystem for Linux!")
        
        # Test with different rate
        bridge.speak_windows("This is faster speech", rate=5)
        bridge.speak_windows("This is slower speech", rate=-5)
        
        # Test with specific voice if available
        if voices and any('Zira' in v for v in voices):
            bridge.speak_windows("Hello, I'm Zira!", voice="Microsoft Zira Desktop")
            
    else:
        print("❌ No PowerShell access - are you in WSL?")


if __name__ == "__main__":
    test_wsl_audio()