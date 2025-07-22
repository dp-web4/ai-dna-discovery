#!/usr/bin/env python3
"""
Check what audio devices are available in WSL vs Windows
"""

import subprocess
import os

def check_wsl_audio():
    """Check WSL audio capabilities"""
    print("=== WSL Audio Device Check ===\n")
    
    # Check ALSA devices
    print("1. ALSA devices in WSL:")
    try:
        result = subprocess.run(['cat', '/proc/asound/cards'], capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("   No ALSA sound cards found")
    except:
        print("   /proc/asound/cards not found")
    
    # Check PulseAudio
    print("\n2. PulseAudio status:")
    try:
        result = subprocess.run(['pulseaudio', '--check'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   PulseAudio is running")
        else:
            print("   PulseAudio is not running")
    except:
        print("   PulseAudio not installed")
    
    # Check if Windows audio is accessible via file system
    print("\n3. Windows audio access:")
    win_paths = [
        '/mnt/c/Windows/System32/drivers',
        '/mnt/c/Windows/System32/mmsys.cpl'
    ]
    
    for path in win_paths:
        if os.path.exists(path):
            print(f"   ✅ {path} accessible")
        else:
            print(f"   ❌ {path} not found")

def check_windows_audio():
    """Check Windows audio devices via PowerShell"""
    print("\n=== Windows Audio Devices ===\n")
    
    ps_script = """
# Get audio devices
Get-WmiObject -Class Win32_SoundDevice | Select-Object Name, Status
Write-Host ""
Write-Host "Audio Endpoints:"
Get-AudioDevice -List | Select-Object Name, Type, Default
"""
    
    try:
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=10)
        
        print("Windows audio devices:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
    except Exception as e:
        print(f"Failed to check Windows audio: {e}")

def check_microphone_permissions():
    """Check microphone permissions"""
    print("\n=== Microphone Permissions ===\n")
    
    ps_script = """
# Check microphone privacy settings
$regPath = "HKCU:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone"
if (Test-Path $regPath) {
    $micAccess = Get-ItemProperty -Path $regPath -Name "Value" -ErrorAction SilentlyContinue
    if ($micAccess) {
        Write-Host "Microphone access: $($micAccess.Value)"
    } else {
        Write-Host "Could not read microphone permissions"
    }
} else {
    Write-Host "Microphone permission registry not found"
}

# List microphone devices
Write-Host ""
Write-Host "Microphone devices:"
Get-WmiObject -Class Win32_SoundDevice | Where-Object {$_.Name -like "*microphone*" -or $_.Name -like "*mic*"} | Select-Object Name, Status
"""
    
    try:
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=10)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
    except Exception as e:
        print(f"Failed to check microphone permissions: {e}")

if __name__ == "__main__":
    check_wsl_audio()
    check_windows_audio() 
    check_microphone_permissions()