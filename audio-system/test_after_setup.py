#!/usr/bin/env python3
"""
Test speech recognition after Windows Speech Recognition setup
"""

import subprocess

def test_post_setup_speech():
    """Test speech recognition after Windows setup"""
    
    ps_script = '''
Add-Type -AssemblyName System.Speech
$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$rec.SetInputToDefaultAudioDevice()

# Simple test words
$words = @("hello", "claude", "test", "computer", "working")
$choices = New-Object System.Speech.Recognition.Choices($words)
$gb = New-Object System.Speech.Recognition.GrammarBuilder($choices)
$grammar = New-Object System.Speech.Recognition.Grammar($gb)
$rec.LoadGrammar($grammar)

Write-Output "READY: Speech recognition test ready"
Write-Output "SAY: Please say 'hello claude' clearly"

# Try recognition with a reasonable timeout
$result = $rec.Recognize([TimeSpan]::FromSeconds(8))

if ($result -and $result.Text.Trim() -ne "") {
    Write-Output "SUCCESS: Heard '$($result.Text)' with confidence $($result.Confidence)"
} else {
    Write-Output "FAILED: No speech detected"
}

$rec.Dispose()
'''
    
    try:
        print("🎤 Testing speech recognition after Windows setup...")
        print("Please say 'hello claude' when prompted...")
        
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=12)
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
        
        return "SUCCESS:" in result.stdout
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Post-Setup Speech Test ===\n")
    
    input("Press Enter after you've completed Windows Speech Recognition setup...")
    
    success = test_post_setup_speech()
    
    if success:
        print("\n🎉 Speech recognition is working!")
        print("✅ Ready to test full conversation system!")
    else:
        print("\n🔧 Still having issues. Let's troubleshoot further.")