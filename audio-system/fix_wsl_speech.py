#!/usr/bin/env python3
"""
Fix WSL speech recognition by trying different approaches
"""

import subprocess
import time

def test_continuous_recognition():
    """Test continuous recognition mode"""
    
    ps_script = '''
Add-Type -AssemblyName System.Speech

$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$rec.SetInputToDefaultAudioDevice()

# Configure for better sensitivity
$rec.InitialSilenceTimeout = [TimeSpan]::FromSeconds(30)
$rec.BabbleTimeout = [TimeSpan]::FromSeconds(5) 
$rec.EndSilenceTimeout = [TimeSpan]::FromSeconds(3)

# Simple word list for testing
$words = @("hello", "test", "claude", "computer", "yes", "no")
$choices = New-Object System.Speech.Recognition.Choices($words)
$gb = New-Object System.Speech.Recognition.GrammarBuilder($choices)
$grammar = New-Object System.Speech.Recognition.Grammar($gb)
$rec.LoadGrammar($grammar)

Write-Output "READY: Say 'hello' or 'test' clearly"

# Try asynchronous recognition
$rec.RecognizeAsync([System.Speech.Recognition.RecognizeMode]::Multiple)

$startTime = Get-Date
$endTime = $startTime.AddSeconds(15)
$resultFound = $false

while ((Get-Date) -lt $endTime -and -not $resultFound) {
    try {
        # Poll for results
        $result = $rec.Recognize([TimeSpan]::FromMilliseconds(100))
        if ($result -and $result.Text.Trim() -ne "") {
            Write-Output "SUCCESS: Heard '$($result.Text)' confidence=$($result.Confidence)"
            $resultFound = $true
        }
    } catch {
        # Continue
    }
    
    $elapsed = ((Get-Date) - $startTime).TotalSeconds
    if ([int]$elapsed % 3 -eq 0 -and [int]($elapsed * 10) % 10 -eq 0) {
        Write-Output "WAITING: $([int]$elapsed)s elapsed, keep speaking..."
    }
    
    Start-Sleep -Milliseconds 100
}

$rec.RecognizeAsyncStop()
$rec.Dispose()

if (-not $resultFound) {
    Write-Output "TIMEOUT: No speech detected in 15 seconds"
}
'''
    
    try:
        print("🎤 Testing continuous speech recognition...")
        print("Say 'hello' or 'test' clearly into your microphone")
        
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=20)
        
        print("\nResults:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
        
        return "SUCCESS:" in result.stdout
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def test_system_speech_recognition():
    """Test Windows built-in speech recognition"""
    
    ps_script = '''
# Try using Windows Speech Recognition directly
$culture = [System.Globalization.CultureInfo]::GetCultureInfo("en-US")
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture)

try {
    # Get all available audio inputs
    Write-Output "=== Available Audio Inputs ==="
    foreach ($device in [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()) {
        Write-Output "Recognizer: $($device.Name) - Culture: $($device.Culture) - Description: $($device.Description)"
    }
    
    $recognizer.SetInputToDefaultAudioDevice()
    Write-Output "INPUT_SET: Default audio device"
    
    # Very simple grammar
    $grammar = New-Object System.Speech.Recognition.DictationGrammar
    $recognizer.LoadGrammar($grammar)
    Write-Output "GRAMMAR_LOADED: Dictation"
    
    # Try a single recognition attempt with longer timeout
    Write-Output "LISTENING: 10 seconds for any speech..."
    $result = $recognizer.Recognize([TimeSpan]::FromSeconds(10))
    
    if ($result) {
        Write-Output "DETECTED: '$($result.Text)' confidence=$($result.Confidence)"
    } else {
        Write-Output "NO_SPEECH: Nothing detected"
    }
    
} catch {
    Write-Output "ERROR: $($_.Exception.Message)"
} finally {
    $recognizer.Dispose()
}
'''
    
    try:
        print("\n🔍 Testing system speech recognition...")
        
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=15)
        
        print("\nSystem test results:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
        
        return "DETECTED:" in result.stdout
        
    except Exception as e:
        print(f"System test failed: {e}")
        return False

def simple_voice_test():
    """Very simple voice detection test"""
    
    print("\n🗣️  Simple voice test:")
    print("Please say 'HELLO CLAUDE' loudly and clearly when prompted...")
    time.sleep(2)
    
    ps_script = '''
Add-Type -AssemblyName System.Speech
$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$rec.SetInputToDefaultAudioDevice()

$words = @("hello", "claude", "hello claude")
$choices = New-Object System.Speech.Recognition.Choices($words)
$gb = New-Object System.Speech.Recognition.GrammarBuilder($choices)
$grammar = New-Object System.Speech.Recognition.Grammar($gb)
$rec.LoadGrammar($grammar)

Write-Output "SAY_NOW: Say 'HELLO CLAUDE' now!"
$result = $rec.Recognize([TimeSpan]::FromSeconds(5))

if ($result) {
    Write-Output "HEARD: $($result.Text)"
} else {
    Write-Output "SILENCE: Nothing heard"
}

$rec.Dispose()
'''
    
    try:
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=10)
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
        
        return "HEARD:" in result.stdout
        
    except Exception as e:
        print(f"Simple test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== WSL Speech Recognition Fixes ===\n")
    
    # Try different approaches
    success1 = test_continuous_recognition()
    success2 = test_system_speech_recognition()  
    success3 = simple_voice_test()
    
    print(f"\n=== Results Summary ===")
    print(f"Continuous recognition: {'✅' if success1 else '❌'}")
    print(f"System recognition: {'✅' if success2 else '❌'}")
    print(f"Simple test: {'✅' if success3 else '❌'}")
    
    if any([success1, success2, success3]):
        print("\n🎉 Speech recognition is working! We can build the conversation system.")
    else:
        print("\n🔧 Speech recognition needs more work. Possible issues:")
        print("   - Microphone input level too low")
        print("   - Wrong default audio device selected")
        print("   - Windows speech recognition not properly configured")
        print("   - Try speaking directly into microphone at normal volume")