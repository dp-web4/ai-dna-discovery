#!/usr/bin/env python3
"""
Minimal WSL microphone test
"""

import subprocess

def test_minimal_mic():
    """Most basic microphone test"""
    ps_script = '''
Add-Type -AssemblyName System.Speech
$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine
try {
    $rec.SetInputToDefaultAudioDevice()
    Write-Output "MIC_ACCESS: SUCCESS"
    
    # Try to detect ANY audio for 3 seconds
    $choices = New-Object System.Speech.Recognition.Choices("hello", "test", "one", "two", "three")
    $gb = New-Object System.Speech.Recognition.GrammarBuilder($choices)
    $grammar = New-Object System.Speech.Recognition.Grammar($gb)
    $rec.LoadGrammar($grammar)
    
    Write-Output "LISTENING: 3 seconds"
    $result = $rec.Recognize([TimeSpan]::FromSeconds(3))
    if ($result) {
        Write-Output "HEARD: $($result.Text)"
    } else {
        Write-Output "HEARD: nothing"
    }
} catch {
    Write-Output "MIC_ACCESS: FAILED - $_"
} finally {
    $rec.Dispose()
}
'''
    
    try:
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=10)
        
        print("Output:")
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
        
        if result.stderr:
            print("Errors:")
            print(f"  {result.stderr}")
            
        return "SUCCESS" in result.stdout
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Minimal WSL Microphone Test ===")
    print("Say 'hello' or 'test' when prompted...")
    
    if test_minimal_mic():
        print("\n✅ Microphone access confirmed!")
    else:
        print("\n❌ Microphone access failed")