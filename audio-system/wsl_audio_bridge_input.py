#!/usr/bin/env python3
"""
WSL Audio Input Bridge - Record via Windows APIs
"""

import subprocess
import tempfile
import os
import time
import wave

class WSLAudioRecorder:
    def __init__(self):
        self.powershell_path = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    
    def record_audio(self, duration_seconds=5, output_file=None):
        """Record audio using Windows PowerShell"""
        
        if output_file is None:
            # Create temp file in Windows-accessible location
            output_file = f"/mnt/c/temp/wsl_recording_{int(time.time())}.wav"
            os.makedirs("/mnt/c/temp", exist_ok=True)
        
        # Convert WSL path to Windows path
        win_output_file = output_file.replace("/mnt/c/", "C:\\")
        
        ps_script = f"""
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Threading;

public class AudioRecorder {{
    [DllImport("winmm.dll")]
    public static extern int mciSendString(string command, string returnString, int returnLength, IntPtr hwndCallback);
    
    public static void RecordAudio(string filename, int durationMs) {{
        string openCommand = "open new type waveaudio alias recsound";
        string recordCommand = "record recsound";
        string saveCommand = $"save recsound {{filename}}";
        string closeCommand = "close recsound";
        
        mciSendString(openCommand, null, 0, IntPtr.Zero);
        mciSendString(recordCommand, null, 0, IntPtr.Zero);
        Thread.Sleep(durationMs);
        mciSendString("stop recsound", null, 0, IntPtr.Zero);
        mciSendString(saveCommand, null, 0, IntPtr.Zero);
        mciSendString(closeCommand, null, 0, IntPtr.Zero);
    }}
}}
"@

Write-Host "Recording audio for {duration_seconds} seconds..."
[AudioRecorder]::RecordAudio("{win_output_file}", {duration_seconds * 1000})
Write-Host "Recording saved to: {win_output_file}"

if (Test-Path "{win_output_file}") {{
    $fileInfo = Get-Item "{win_output_file}"
    Write-Host "File size: $($fileInfo.Length) bytes"
}} else {{
    Write-Host "ERROR: Recording file not created"
}}
"""
        
        try:
            print(f"🎤 Recording for {duration_seconds} seconds...")
            print(f"📁 Output file: {output_file}")
            
            result = subprocess.run([
                self.powershell_path,
                '-Command', ps_script
            ], capture_output=True, text=True, timeout=duration_seconds + 10)
            
            print("PowerShell output:", result.stdout)
            if result.stderr:
                print("PowerShell errors:", result.stderr)
            
            # Check if file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✅ Recording successful! File size: {file_size} bytes")
                return output_file
            else:
                print("❌ Recording failed - file not found")
                return None
                
        except subprocess.TimeoutExpired:
            print("⏰ Recording timed out")
            return None
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return None

def test_audio_recording():
    """Test audio recording functionality"""
    recorder = WSLAudioRecorder()
    
    print("=== WSL Audio Recording Test ===")
    print("Speak into your microphone...")
    
    recorded_file = recorder.record_audio(5)
    
    if recorded_file:
        print(f"🎉 Successfully recorded audio to: {recorded_file}")
        
        # Try to get basic info about the recording
        try:
            if recorded_file.endswith('.wav'):
                with wave.open(recorded_file, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    duration = frames / sample_rate
                    
                    print(f"📊 Audio info:")
                    print(f"   Duration: {duration:.2f} seconds")
                    print(f"   Sample rate: {sample_rate} Hz")
                    print(f"   Channels: {channels}")
                    print(f"   Frames: {frames}")
        except Exception as e:
            print(f"Could not read audio file info: {e}")
    else:
        print("❌ Recording failed")

if __name__ == "__main__":
    test_audio_recording()