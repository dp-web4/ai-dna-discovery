#!/usr/bin/env python3
"""
Test actual audio input levels to see if microphone is working
"""

import subprocess

def test_windows_audio_levels():
    """Test if we can detect audio input levels"""
    
    ps_script = '''
# Try to access Windows audio APIs more directly
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public class AudioLevels {
    [DllImport("winmm.dll")]
    public static extern int waveInGetNumDevs();
    
    [DllImport("winmm.dll")]
    public static extern int waveInGetDevCaps(int deviceId, ref WAVEINCAPS caps, int size);
    
    [StructLayout(LayoutKind.Sequential)]
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
}
"@

# Check available input devices
$deviceCount = [AudioLevels]::waveInGetNumDevs()
Write-Output "AUDIO_DEVICES: Found $deviceCount input devices"

for ($i = 0; $i -lt $deviceCount; $i++) {
    $caps = New-Object AudioLevels+WAVEINCAPS
    $result = [AudioLevels]::waveInGetDevCaps($i, [ref]$caps, [System.Runtime.InteropServices.Marshal]::SizeOf($caps))
    if ($result -eq 0) {
        Write-Output "DEVICE_$i`: $($caps.szPname)"
    }
}

# Test microphone accessibility through different method
Write-Output ""
Write-Output "TESTING: Basic microphone test..."

# Very basic audio test
try {
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.Speak("Testing microphone access")
    $synth.Dispose()
    Write-Output "TTS_TEST: Working"
} catch {
    Write-Output "TTS_TEST: Failed - $($_.Exception.Message)"
}
'''
    
    try:
        print("🔍 Testing Windows audio device levels...")
        
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=10)
        
        print("Audio device results:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
        
        if result.stderr:
            print("\nErrors:")
            for line in result.stderr.strip().split('\n'):
                if line.strip():
                    print(f"  {line}")
        
    except Exception as e:
        print(f"Audio test failed: {e}")

def check_microphone_permissions_detailed():
    """Check microphone permissions in detail"""
    
    ps_script = '''
Write-Output "=== Microphone Permission Check ==="

# Check global microphone access
$globalAccess = Get-ItemProperty -Path "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone" -Name "Value" -ErrorAction SilentlyContinue
if ($globalAccess) {
    Write-Output "Global mic access: $($globalAccess.Value)"
} else {
    Write-Output "Global mic access: Cannot read"
}

# Check PowerShell specific access
$psAccess = Get-ItemProperty -Path "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone\Microsoft.PowerShell_*" -Name "Value" -ErrorAction SilentlyContinue
if ($psAccess) {
    Write-Output "PowerShell mic access: $($psAccess.Value)"
} else {
    Write-Output "PowerShell mic access: No specific entry found (using global)"
}

# Check if Windows Speech Recognition is enabled
$speechEnabled = Get-ItemProperty -Path "HKCU:\SOFTWARE\Microsoft\Speech_OneCore\Settings\OnlineSpeechPrivacy" -Name "HasAccepted" -ErrorAction SilentlyContinue
if ($speechEnabled) {
    Write-Output "Windows Speech Recognition: $($speechEnabled.HasAccepted)"
} else {
    Write-Output "Windows Speech Recognition: Setting not found"
}

Write-Output ""
Write-Output "=== Current Audio Settings ==="

# Get default recording device
try {
    $audioDevices = Get-CimInstance -ClassName Win32_SoundDevice | Where-Object {$_.Status -eq "OK"}
    foreach ($device in $audioDevices) {
        Write-Output "Audio device: $($device.Name)"
    }
} catch {
    Write-Output "Could not enumerate audio devices"
}
'''
    
    try:
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=10)
        
        print("\n🔧 Detailed permission check:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
        
    except Exception as e:
        print(f"Permission check failed: {e}")

def suggest_solutions():
    """Suggest solutions based on findings"""
    
    print("\n💡 Suggested Solutions:")
    print()
    print("1. **Check Windows microphone settings:**")
    print("   - Open Windows Settings → Privacy → Microphone")
    print("   - Ensure 'Allow apps to access your microphone' is ON")
    print("   - Ensure 'Allow desktop apps to access your microphone' is ON")
    print()
    print("2. **Check default recording device:**")
    print("   - Right-click speaker icon in taskbar → 'Recording devices'")
    print("   - Ensure correct microphone is set as default")
    print("   - Test microphone levels in Windows")
    print()
    print("3. **Try Windows Speech Recognition setup:**")
    print("   - Type 'speech recognition' in Windows search")
    print("   - Run 'Set up microphone' wizard")
    print("   - Train speech recognition")
    print()
    print("4. **Alternative approach:**")
    print("   - We could use a different audio capture method")
    print("   - Try ffmpeg or other audio tools")
    print("   - Use file-based audio exchange")

if __name__ == "__main__":
    print("=== WSL Audio Input Investigation ===\n")
    
    test_windows_audio_levels()
    check_microphone_permissions_detailed()
    suggest_solutions()
    
    print("\n🎯 Next steps:")
    print("   1. Check your Windows microphone settings")
    print("   2. Test microphone in Windows first")
    print("   3. Then we can retry the WSL speech recognition")