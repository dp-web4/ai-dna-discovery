#!/usr/bin/env python3
"""
Raw audio capture from Windows microphone via WSL
"""

import subprocess
import os
import time

def test_raw_audio_recording():
    """Use Windows SoundRecorder or media APIs for raw recording"""
    
    # Create temp directory
    os.makedirs("/mnt/c/temp", exist_ok=True)
    
    ps_script = '''
# Use Windows Media Foundation for raw audio recording
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.IO;

public class RawAudioCapture {
    [DllImport("winmm.dll")]
    public static extern int mciSendString(string command, System.Text.StringBuilder returnString, int returnLength, IntPtr hwndCallback);
    
    public static bool RecordAudio(string filename, int durationMs) {
        try {
            string openCmd = "open new type waveaudio alias recsound";
            string recordCmd = "record recsound";
            string stopCmd = "stop recsound";
            string saveCmd = $"save recsound {filename}";
            string closeCmd = "close recsound";
            
            mciSendString(openCmd, null, 0, IntPtr.Zero);
            mciSendString(recordCmd, null, 0, IntPtr.Zero);
            System.Threading.Thread.Sleep(durationMs);
            mciSendString(stopCmd, null, 0, IntPtr.Zero);
            mciSendString(saveCmd, null, 0, IntPtr.Zero);
            mciSendString(closeCmd, null, 0, IntPtr.Zero);
            
            return File.Exists(filename);
        } catch {
            return false;
        }
    }
}
"@

$outputFile = "C:\\temp\\raw_recording.wav"
$duration = 3000  # 3 seconds

Write-Output "RECORDING: Starting 3-second raw audio capture..."
$success = [RawAudioCapture]::RecordAudio($outputFile, $duration)

if ($success) {
    $fileInfo = Get-Item $outputFile -ErrorAction SilentlyContinue
    if ($fileInfo) {
        Write-Output "SUCCESS: Recorded $($fileInfo.Length) bytes to $outputFile"
    } else {
        Write-Output "FAILED: File not created"
    }
} else {
    Write-Output "FAILED: Recording failed"
}
'''
    
    try:
        print("🎤 Testing raw audio recording...")
        print("Speak into your microphone for 3 seconds...")
        
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=10)
        
        print("Recording result:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        # Check if file was created
        wav_file = "/mnt/c/temp/raw_recording.wav"
        if os.path.exists(wav_file):
            file_size = os.path.getsize(wav_file)
            print(f"✅ Raw recording successful! File size: {file_size} bytes")
            return wav_file
        else:
            print("❌ No recording file created")
            return None
            
    except Exception as e:
        print(f"Raw recording failed: {e}")
        return None

def test_ffmpeg_recording():
    """Try using ffmpeg for direct audio capture"""
    
    # Check if ffmpeg is available
    ps_script_check = '''
$ffmpegPaths = @(
    "C:\\ffmpeg\\bin\\ffmpeg.exe",
    "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"
)

foreach ($path in $ffmpegPaths) {
    if (Test-Path $path) {
        Write-Output "FFMPEG_FOUND: $path"
        break
    }
}

# Try to find in PATH
try {
    $wherePath = where.exe ffmpeg 2>$null
    if ($wherePath) {
        Write-Output "FFMPEG_PATH: $wherePath"
    }
} catch {
    Write-Output "FFMPEG_NOT_IN_PATH"
}
'''
    
    try:
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script_check
        ], capture_output=True, text=True, timeout=5)
        
        ffmpeg_path = None
        for line in result.stdout.strip().split('\n'):
            if "FFMPEG_FOUND:" in line or "FFMPEG_PATH:" in line:
                ffmpeg_path = line.split(": ")[1].strip()
                break
        
        if ffmpeg_path:
            print(f"✅ Found ffmpeg at: {ffmpeg_path}")
            
            # Convert to WSL path
            wsl_ffmpeg = ffmpeg_path.replace('C:', '/mnt/c').replace('\\', '/')
            
            # Try recording with ffmpeg
            print("🎤 Testing ffmpeg audio capture...")
            output_file = "/mnt/c/temp/ffmpeg_recording.wav"
            
            cmd = [
                wsl_ffmpeg,
                '-f', 'dshow',
                '-i', 'audio="Microphone Array (Realtek(R) Audio)"',
                '-t', '3',
                '-y',
                output_file
            ]
            
            print("Recording 3 seconds with ffmpeg...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✅ FFmpeg recording successful! File size: {file_size} bytes")
                return output_file
            else:
                print("❌ FFmpeg recording failed")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return None
        else:
            print("❌ FFmpeg not found")
            return None
            
    except Exception as e:
        print(f"FFmpeg test failed: {e}")
        return None

def test_direct_winapi():
    """Test direct Windows API audio capture"""
    
    ps_script = '''
# Direct WinAPI approach
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public class DirectAudio {
    [DllImport("winmm.dll")]
    public static extern int waveInGetNumDevs();
    
    [DllImport("winmm.dll", CharSet = CharSet.Auto)]
    public static extern int waveInGetDevCaps(int deviceId, ref WAVEINCAPS pwic, int cbwic);
    
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public struct WAVEINCAPS {
        public short wMid;
        public short wPid;
        public int vDriverVersion;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
        public string szPname;
        public int dwFormats;
        public short wChannels;
        public short wReserved1;
    }
    
    public static void ListDevices() {
        int deviceCount = waveInGetNumDevs();
        Console.WriteLine($"Found {deviceCount} audio input devices:");
        
        for (int i = 0; i < deviceCount; i++) {
            WAVEINCAPS caps = new WAVEINCAPS();
            if (waveInGetDevCaps(i, ref caps, Marshal.SizeOf(caps)) == 0) {
                Console.WriteLine($"Device {i}: {caps.szPname} (Channels: {caps.wChannels})");
            }
        }
    }
}
"@

Write-Output "=== Direct Audio Device Access ==="
[DirectAudio]::ListDevices()
'''
    
    try:
        print("\n🔍 Testing direct Windows audio API access...")
        
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=5)
        
        print("Direct API results:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
        
    except Exception as e:
        print(f"Direct API test failed: {e}")

if __name__ == "__main__":
    print("=== Raw Audio Capture Investigation ===\n")
    
    # Try multiple approaches for raw audio access
    wav_file1 = test_raw_audio_recording()
    wav_file2 = test_ffmpeg_recording()
    test_direct_winapi()
    
    print(f"\n=== Results ===")
    if wav_file1 or wav_file2:
        print("✅ Raw audio capture is working!")
        print("We can build a conversation system using direct audio recording!")
        
        working_file = wav_file1 or wav_file2
        print(f"📁 Audio file: {working_file}")
        
    else:
        print("🔧 Raw audio capture needs more work")
        print("Let's try alternative approaches...")
        
    print(f"\n🎯 Next step: Build conversation system with working audio method")