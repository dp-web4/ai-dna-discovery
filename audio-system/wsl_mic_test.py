#!/usr/bin/env python3
"""
Test WSL microphone access via PowerShell
"""

import subprocess
import tempfile
import os
import time

def test_powershell_mic():
    """Test if we can record audio via PowerShell"""
    print("Testing PowerShell microphone access...")
    
    # Create temp file for recording
    temp_file = tempfile.mktemp(suffix=".wav")
    
    # PowerShell script to record audio
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$waveform = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$recognizer.SetInputToDefaultAudioDevice()
Write-Host "Recording for 3 seconds..."
Start-Sleep -Seconds 3
Write-Host "Recording complete"
"""
    
    try:
        # Test basic PowerShell audio access
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=10)
        
        print("PowerShell output:", result.stdout)
        print("PowerShell errors:", result.stderr)
        
    except Exception as e:
        print(f"PowerShell test failed: {e}")

def test_windows_speech_recognition():
    """Test Windows Speech Recognition API"""
    print("\nTesting Windows Speech Recognition API...")
    
    ps_script = """
Add-Type -AssemblyName System.Speech
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$recognizer.SetInputToDefaultAudioDevice()
$grammar = New-Object System.Speech.Recognition.DictationGrammar
$recognizer.LoadGrammar($grammar)
$recognizer.RecognizeMode = [System.Speech.Recognition.RecognizeMode]::Single
Write-Host "Listening for speech..."
$result = $recognizer.Recognize([System.TimeSpan]::FromSeconds(5))
if ($result) {
    Write-Host "Recognized: $($result.Text)"
} else {
    Write-Host "No speech detected"
}
$recognizer.Dispose()
"""
    
    try:
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=15)
        
        print("Speech recognition output:", result.stdout)
        print("Speech recognition errors:", result.stderr)
        
    except Exception as e:
        print(f"Speech recognition test failed: {e}")

if __name__ == "__main__":
    test_powershell_mic()
    test_windows_speech_recognition()