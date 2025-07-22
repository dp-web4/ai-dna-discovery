#!/usr/bin/env python3
"""
WSL Speech Recognition via Windows Speech API
"""

import subprocess
import json
import time

class WSLSpeechRecognizer:
    def __init__(self):
        self.powershell_path = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    
    def listen_for_speech(self, timeout_seconds=5):
        """Listen for speech using Windows Speech Recognition"""
        
        ps_script = f"""
Add-Type -AssemblyName System.Speech
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$recognizer.SetInputToDefaultAudioDevice()

# Create grammar for better recognition
$choices = @('hello', 'test', 'sprout', 'claude', 'consciousness', 'audio', 'system')
$grammarBuilder = New-Object System.Speech.Recognition.GrammarBuilder
$grammarBuilder.Append((New-Object System.Speech.Recognition.Choices($choices)))
$grammar = New-Object System.Speech.Recognition.Grammar($grammarBuilder)

# Also load dictation grammar
$dictationGrammar = New-Object System.Speech.Recognition.DictationGrammar
$recognizer.LoadGrammar($grammar)
$recognizer.LoadGrammar($dictationGrammar)

Write-Host "Listening for {timeout_seconds} seconds..."
$recognizer.RecognizeAsync([System.Speech.Recognition.RecognizeMode]::Multiple)

$endTime = (Get-Date).AddSeconds({timeout_seconds})
$results = @()

while ((Get-Date) -lt $endTime) {{
    $result = $recognizer.Recognize([System.TimeSpan]::FromMilliseconds(100))
    if ($result -and $result.Text) {{
        $results += $result.Text
        Write-Host "Heard: $($result.Text) (Confidence: $($result.Confidence))"
    }}
    Start-Sleep -Milliseconds 100
}}

$recognizer.RecognizeAsyncStop()
$recognizer.Dispose()

if ($results.Count -gt 0) {{
    Write-Host "FINAL_RESULTS: $($results -join ' | ')"
}} else {{
    Write-Host "FINAL_RESULTS: SILENCE"
}}
"""
        
        try:
            print(f"🎤 Listening for {timeout_seconds} seconds...")
            result = subprocess.run([
                self.powershell_path,
                '-Command', ps_script
            ], capture_output=True, text=True, timeout=timeout_seconds + 5)
            
            # Parse results
            output_lines = result.stdout.strip().split('\n')
            final_results = None
            
            for line in output_lines:
                print(f"PS: {line}")
                if line.startswith("FINAL_RESULTS:"):
                    final_results = line.replace("FINAL_RESULTS:", "").strip()
            
            if final_results and final_results != "SILENCE":
                return final_results
            else:
                return None
                
        except subprocess.TimeoutExpired:
            print("⏰ Speech recognition timed out")
            return None
        except Exception as e:
            print(f"❌ Speech recognition failed: {e}")
            return None

def test_wsl_speech():
    """Test WSL speech recognition"""
    recognizer = WSLSpeechRecognizer()
    
    print("=== WSL Speech Recognition Test ===")
    print("Say something like 'hello', 'test', or 'sprout'...")
    
    result = recognizer.listen_for_speech(10)
    
    if result:
        print(f"✅ Recognized: '{result}'")
    else:
        print("❌ No speech detected")

if __name__ == "__main__":
    test_wsl_speech()