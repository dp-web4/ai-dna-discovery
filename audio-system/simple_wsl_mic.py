#!/usr/bin/env python3
"""
Simple WSL microphone access using Windows SoundRecorder
"""

import subprocess
import tempfile
import os
import time

def test_sound_recorder():
    """Test using Windows SoundRecorder.exe"""
    print("=== Testing Windows SoundRecorder ===")
    
    # Check if SoundRecorder exists
    sound_recorder_paths = [
        '/mnt/c/Windows/System32/SoundRecorder.exe',
        '/mnt/c/Windows/SysWOW64/SoundRecorder.exe'
    ]
    
    recorder_path = None
    for path in sound_recorder_paths:
        if os.path.exists(path):
            recorder_path = path
            print(f"✅ Found SoundRecorder at: {path}")
            break
    
    if not recorder_path:
        print("❌ SoundRecorder.exe not found")
        return False
    
    # Try to run it (this might open a GUI)
    try:
        print("Attempting to run SoundRecorder...")
        result = subprocess.run([recorder_path, '/?'], capture_output=True, text=True, timeout=5)
        print("SoundRecorder output:", result.stdout)
        print("SoundRecorder errors:", result.stderr)
    except Exception as e:
        print(f"SoundRecorder test failed: {e}")

def test_ffmpeg_windows():
    """Test using ffmpeg for Windows (if available)"""
    print("\n=== Testing FFmpeg for audio capture ===")
    
    # Check common ffmpeg locations
    ffmpeg_paths = [
        '/mnt/c/ffmpeg/bin/ffmpeg.exe',
        '/mnt/c/Program Files/ffmpeg/bin/ffmpeg.exe',
        '/mnt/c/Users/admin/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-7.0-full_build/bin/ffmpeg.exe'
    ]
    
    ffmpeg_path = None
    for path in ffmpeg_paths:
        if os.path.exists(path):
            ffmpeg_path = path
            print(f"✅ Found ffmpeg at: {path}")
            break
    
    if not ffmpeg_path:
        print("❌ ffmpeg not found in common locations")
        
        # Try to find ffmpeg in PATH
        try:
            result = subprocess.run(['where.exe', 'ffmpeg'], capture_output=True, text=True)
            if result.returncode == 0:
                win_path = result.stdout.strip()
                # Convert Windows path to WSL path
                if win_path.startswith('C:'):
                    ffmpeg_path = win_path.replace('C:', '/mnt/c').replace('\\', '/')
                    print(f"✅ Found ffmpeg in PATH: {ffmpeg_path}")
        except:
            pass
    
    if ffmpeg_path:
        # Test ffmpeg audio devices
        try:
            print("Listing audio input devices...")
            result = subprocess.run([
                ffmpeg_path, '-f', 'dshow', '-list_devices', 'true', '-i', 'dummy'
            ], capture_output=True, text=True, timeout=10)
            
            print("FFmpeg device list:")
            print(result.stderr)  # ffmpeg outputs device info to stderr
            
        except Exception as e:
            print(f"FFmpeg device list failed: {e}")
    
    return ffmpeg_path

def test_powershell_simple_record():
    """Simple PowerShell recording using Windows Media APIs"""
    print("\n=== Testing Simple PowerShell Recording ===")
    
    ps_script = """
# Simple recording using Windows Media Format SDK
$duration = 3
$outputFile = "C:\\temp\\simple_recording.wav"

# Create temp directory
New-Item -ItemType Directory -Force -Path "C:\\temp" | Out-Null

# Use Windows Speech API for simple recording
Add-Type -AssemblyName System.Speech
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine

try {
    $recognizer.SetInputToDefaultAudioDevice()
    Write-Host "Microphone access successful"
    
    # Simple presence detection
    Write-Host "Testing microphone for $duration seconds..."
    $endTime = (Get-Date).AddSeconds($duration)
    $detected = $false
    
    while ((Get-Date) -lt $endTime) {
        try {
            $result = $recognizer.Recognize([System.TimeSpan]::FromMilliseconds(100))
            if ($result) {
                Write-Host "Audio detected: $($result.Text)"
                $detected = $true
            }
        } catch {
            # Ignore recognition errors, just check if mic is working
        }
        Start-Sleep -Milliseconds 100
    }
    
    if ($detected) {
        Write-Host "SUCCESS: Microphone is working and detected audio"
    } else {
        Write-Host "INFO: Microphone accessible but no clear speech detected"
    }
    
} catch {
    Write-Host "ERROR: Could not access microphone - $_"
} finally {
    $recognizer.Dispose()
}
"""
    
    try:
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=15)
        
        print("PowerShell recording test:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        return "SUCCESS" in result.stdout
        
    except Exception as e:
        print(f"PowerShell recording test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎤 WSL Microphone Access Investigation\n")
    
    test_sound_recorder()
    ffmpeg_found = test_ffmpeg_windows()
    mic_working = test_powershell_simple_record()
    
    print(f"\n=== Summary ===")
    print(f"FFmpeg available: {'Yes' if ffmpeg_found else 'No'}")
    print(f"Microphone accessible: {'Yes' if mic_working else 'Unknown'}")
    
    if ffmpeg_found and mic_working:
        print("✅ WSL microphone access should be possible!")
    elif mic_working:
        print("⚡ Microphone works via PowerShell - we can build on this")
    else:
        print("❌ Need to investigate further")