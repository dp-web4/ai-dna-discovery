#!/usr/bin/env python3
"""
Debug WSL speech recognition issues
"""

import subprocess
import time

def test_microphone_levels():
    """Test if we can detect any audio input levels"""
    
    ps_script = '''
Add-Type -AssemblyName System.Speech
$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine

try {
    $rec.SetInputToDefaultAudioDevice()
    Write-Output "MIC_INITIALIZED: SUCCESS"
    
    # Test with very permissive settings
    $rec.InitialSilenceTimeout = [TimeSpan]::FromSeconds(1)
    $rec.BabbleTimeout = [TimeSpan]::FromSeconds(0)
    $rec.EndSilenceTimeout = [TimeSpan]::FromSeconds(1)
    
    # Create a very broad grammar
    $words = @("test", "hello", "one", "two", "three", "four", "five", "audio", "sound", "voice", "mic", "microphone", "claude", "computer")
    $choices = New-Object System.Speech.Recognition.Choices($words)
    $gb = New-Object System.Speech.Recognition.GrammarBuilder($choices)
    $grammar = New-Object System.Speech.Recognition.Grammar($gb)
    $rec.LoadGrammar($grammar)
    
    Write-Output "GRAMMAR_LOADED: SUCCESS"
    
    # Also try dictation
    $dictation = New-Object System.Speech.Recognition.DictationGrammar
    $rec.LoadGrammar($dictation)
    Write-Output "DICTATION_LOADED: SUCCESS"
    
    Write-Output "LISTENING_VERBOSE: Starting 10 second test"
    $startTime = Get-Date
    $endTime = $startTime.AddSeconds(10)
    $attemptCount = 0
    
    while ((Get-Date) -lt $endTime) {
        $attemptCount++
        try {
            # Very short timeout to be responsive
            $result = $rec.Recognize([TimeSpan]::FromMilliseconds(200))
            if ($result) {
                Write-Output "DETECTED: '$($result.Text)' confidence=$($result.Confidence) attempt=$attemptCount"
            }
        } catch [System.InvalidOperationException] {
            Write-Output "EXCEPTION: InvalidOperation at attempt $attemptCount"
        } catch [System.TimeoutException] {
            # Normal timeout, continue
        } catch {
            Write-Output "EXCEPTION: $($_.Exception.GetType().Name) - $($_.Exception.Message)"
        }
        
        if ($attemptCount % 50 -eq 0) {
            Write-Output "HEARTBEAT: attempt $attemptCount"
        }
        
        Start-Sleep -Milliseconds 20
    }
    
    Write-Output "LISTENING_COMPLETE: Total attempts $attemptCount"
    
} catch {
    Write-Output "INITIALIZATION_FAILED: $($_.Exception.Message)"
} finally {
    if ($rec) {
        $rec.Dispose()
        Write-Output "CLEANUP: Complete"
    }
}
'''
    
    try:
        print("🔍 Running detailed microphone diagnostics...")
        print("Please speak clearly into your microphone during the test.")
        
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=15)
        
        print("\n📊 Diagnostic Results:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
        
        if result.stderr:
            print("\n❌ Errors:")
            for line in result.stderr.strip().split('\n'):
                if line.strip():
                    print(f"  {line}")
        
        # Analyze results
        output = result.stdout
        if "DETECTED:" in output:
            print("\n✅ Speech detection is working!")
            return True
        elif "MIC_INITIALIZED: SUCCESS" in output:
            print("\n⚠️  Microphone accessible but no speech detected")
            print("   Try speaking louder or check microphone settings")
            return False
        else:
            print("\n❌ Microphone initialization failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Test timed out")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def check_windows_microphone_settings():
    """Check Windows microphone settings"""
    
    ps_script = '''
# Check microphone privacy settings
Write-Output "=== Microphone Privacy Settings ==="
$microphoneAccess = Get-ItemProperty -Path "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone" -Name "Value" -ErrorAction SilentlyContinue
if ($microphoneAccess) {
    Write-Output "Global microphone access: $($microphoneAccess.Value)"
} else {
    Write-Output "Could not read microphone privacy settings"
}

# Check default audio devices
Write-Output ""
Write-Output "=== Audio Devices ==="
try {
    $devices = Get-WmiObject -Class Win32_SoundDevice | Where-Object {$_.Status -eq "OK"}
    foreach ($device in $devices) {
        Write-Output "Device: $($device.Name) - Status: $($device.Status)"
    }
} catch {
    Write-Output "Could not enumerate audio devices"
}

# Check for speech recognition settings
Write-Output ""
Write-Output "=== Speech Recognition ==="
$speechRegPath = "HKCU:\SOFTWARE\Microsoft\Speech"
if (Test-Path $speechRegPath) {
    Write-Output "Speech recognition registry found"
} else {
    Write-Output "No speech recognition registry found"
}
'''
    
    try:
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=10)
        
        print("🔧 Windows Microphone Settings:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
                
    except Exception as e:
        print(f"Settings check failed: {e}")

if __name__ == "__main__":
    print("=== WSL Speech Recognition Debug ===\n")
    
    check_windows_microphone_settings()
    print("\n" + "="*50 + "\n")
    
    success = test_microphone_levels()
    
    if success:
        print("\n🎉 Microphone is working! The conversation system should work.")
    else:
        print("\n🔧 Microphone needs attention. Possible solutions:")
        print("   1. Check Windows microphone privacy settings")
        print("   2. Ensure microphone is not muted")
        print("   3. Try speaking louder or closer to microphone")
        print("   4. Check if another application is using the microphone")