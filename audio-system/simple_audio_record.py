#!/usr/bin/env python3
"""
Simple audio recording using Windows built-in tools
"""

import subprocess
import os
import time

def try_sox_recording():
    """Try using SoX if available"""
    
    # Check for SoX
    ps_script = '''
$soxPaths = @(
    "C:\\sox\\sox.exe",
    "C:\\Program Files\\sox\\sox.exe",
    "C:\\Program Files (x86)\\sox\\sox.exe"
)

foreach ($path in $soxPaths) {
    if (Test-Path $path) {
        Write-Output "SOX_FOUND: $path"
        break
    }
}
'''
    
    try:
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=5)
        
        if "SOX_FOUND:" in result.stdout:
            print("✅ SoX found - could use for recording")
        else:
            print("❌ SoX not available")
            
    except Exception as e:
        print(f"SoX check failed: {e}")

def try_simple_mci_record():
    """Try the simplest possible MCI recording"""
    
    ps_script = '''
# Simple MCI commands without complex C# code
$outputFile = "C:\\temp\\simple_test.wav"

# Create temp directory
New-Item -ItemType Directory -Force -Path "C:\\temp" | Out-Null

Write-Output "STARTING: Simple MCI recording test"

# Use PowerShell to call mciSendString directly
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public class SimpleMCI {
    [DllImport("winmm.dll", CharSet = CharSet.Auto)]
    public static extern int mciSendString(string lpstrCommand, StringBuilder lpstrReturnString, int uReturnLength, IntPtr hwndCallback);
}
"@

try {
    # Open recording device
    $result1 = [SimpleMCI]::mciSendString("open new type waveaudio alias rec", $null, 0, [IntPtr]::Zero)
    Write-Output "OPEN_RESULT: $result1"
    
    if ($result1 -eq 0) {
        # Start recording
        $result2 = [SimpleMCI]::mciSendString("record rec", $null, 0, [IntPtr]::Zero)
        Write-Output "RECORD_RESULT: $result2"
        
        if ($result2 -eq 0) {
            Write-Output "RECORDING: 3 seconds..."
            Start-Sleep -Seconds 3
            
            # Stop recording
            $result3 = [SimpleMCI]::mciSendString("stop rec", $null, 0, [IntPtr]::Zero)
            Write-Output "STOP_RESULT: $result3"
            
            # Save file
            $result4 = [SimpleMCI]::mciSendString("save rec `"$outputFile`"", $null, 0, [IntPtr]::Zero)
            Write-Output "SAVE_RESULT: $result4"
            
            # Close
            [SimpleMCI]::mciSendString("close rec", $null, 0, [IntPtr]::Zero)
            
            # Check if file exists
            if (Test-Path $outputFile) {
                $fileInfo = Get-Item $outputFile
                Write-Output "SUCCESS: File created - $($fileInfo.Length) bytes"
            } else {
                Write-Output "FAILED: No file created"
            }
        }
    }
} catch {
    Write-Output "ERROR: $($_.Exception.Message)"
}
'''
    
    try:
        print("🎤 Testing simple MCI recording...")
        print("Speak into microphone for 3 seconds...")
        
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=10)
        
        print("MCI Recording results:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
        
        if result.stderr:
            print("Errors:")
            for line in result.stderr.strip().split('\n')[:3]:  # Limit error output
                if line.strip():
                    print(f"  {line}")
        
        # Check for the file
        test_file = "/mnt/c/temp/simple_test.wav"
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            print(f"✅ SUCCESS! Audio file created: {file_size} bytes")
            return test_file
        else:
            print("❌ No audio file created")
            return None
            
    except Exception as e:
        print(f"MCI recording failed: {e}")
        return None

def try_command_line_tools():
    """Try Windows command line audio tools"""
    
    print("\n🔍 Checking for Windows audio command line tools...")
    
    # Check for various audio tools
    tools_to_check = [
        'ffmpeg.exe',
        'sox.exe', 
        'audiocap.exe',
        'soundrecorder.exe'
    ]
    
    for tool in tools_to_check:
        try:
            result = subprocess.run(['where.exe', tool], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Found: {tool} at {result.stdout.strip()}")
            else:
                print(f"❌ Not found: {tool}")
        except:
            print(f"❌ Cannot check: {tool}")

def suggest_alternative_approach():
    """Suggest alternative approaches for audio capture"""
    
    print("\n💡 Alternative Approaches:")
    print()
    print("1. **Install FFmpeg for Windows:**")
    print("   - Download from https://ffmpeg.org/download.html")
    print("   - Extract to C:\\ffmpeg\\")
    print("   - Add to PATH or use direct path")
    print()
    print("2. **Use WSL2 with PulseAudio:**")
    print("   - Install PulseAudio in WSL2")
    print("   - Configure Windows audio forwarding")
    print("   - Direct audio access through Linux")
    print()
    print("3. **Text-based conversation for now:**")
    print("   - Skip audio input temporarily")
    print("   - Use text input with voice output")
    print("   - Focus on LLM integration")
    print()
    print("4. **Switch to Sprout for voice conversation:**")
    print("   - Full audio stack is working there")
    print("   - Return to WSL for other features")

if __name__ == "__main__":
    print("=== Simple Audio Recording Test ===\n")
    
    try_sox_recording()
    audio_file = try_simple_mci_record()
    try_command_line_tools()
    
    if audio_file:
        print(f"\n🎉 Audio recording is working!")
        print(f"📁 File: {audio_file}")
        print("✅ We can build a file-based conversation system!")
    else:
        print(f"\n🔧 Audio recording still needs work")
        suggest_alternative_approach()
        
        print(f"\n🎯 Recommendation:")
        print("Let's implement a hybrid approach:")
        print("- ✅ Voice OUTPUT working (you can hear me)")
        print("- 🔄 Text INPUT for now (you can type to me)")
        print("- 🎯 Full voice conversation on Sprout")
        
        print(f"\nWant to try a text+voice conversation? I can speak to you and you can type back!")