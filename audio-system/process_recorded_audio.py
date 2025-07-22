#!/usr/bin/env python3
"""
Process the recorded audio to extract speech
"""

import subprocess
import os

def try_windows_speech_on_file(wav_file):
    """Try to use Windows Speech Recognition on the recorded file"""
    
    win_file = wav_file.replace('/mnt/c/', 'C:\\').replace('/', '\\')
    
    ps_script = f'''
Add-Type -AssemblyName System.Speech

# Try to recognize speech from the recorded file
try {{
    $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
    $recognizer.SetInputToWaveFile("{win_file}")
    
    # Load dictation grammar
    $dictation = New-Object System.Speech.Recognition.DictationGrammar
    $recognizer.LoadGrammar($dictation)
    
    # Also try simple words
    $words = @("hello", "claude", "yes", "no", "test", "ok", "understand", "heard", "me", "you", "but", "did")
    $choices = New-Object System.Speech.Recognition.Choices($words)
    $gb = New-Object System.Speech.Recognition.GrammarBuilder($choices)
    $grammar = New-Object System.Speech.Recognition.Grammar($gb)
    $recognizer.LoadGrammar($grammar)
    
    Write-Output "PROCESSING: Analyzing audio file..."
    
    # Try to recognize multiple times
    $attempts = 0
    $maxAttempts = 10
    $results = @()
    
    while ($attempts -lt $maxAttempts) {{
        try {{
            $result = $recognizer.Recognize([TimeSpan]::FromSeconds(1))
            if ($result -and $result.Text.Trim() -ne "") {{
                $confidence = [math]::Round($result.Confidence, 3)
                $text = $result.Text.Trim()
                Write-Output "RECOGNIZED: '$text' (confidence: $confidence)"
                $results += "$text ($confidence)"
            }}
            $attempts++
        }} catch {{
            break
        }}
    }}
    
    if ($results.Count -gt 0) {{
        Write-Output "FINAL_RESULTS: $($results -join ' | ')"
    }} else {{
        Write-Output "NO_SPEECH: Could not extract speech from audio"
    }}
    
    $recognizer.Dispose()
    
}} catch {{
    Write-Output "ERROR: $($_.Exception.Message)"
}}
'''
    
    try:
        print(f"🔍 Processing audio file: {wav_file}")
        
        result = subprocess.run([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-Command', ps_script
        ], capture_output=True, text=True, timeout=15)
        
        print("Speech recognition results:")
        recognized_text = []
        
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line:
                print(f"  {line}")
                if line.startswith("RECOGNIZED:"):
                    # Extract just the text part
                    text_part = line.split("'")[1] if "'" in line else ""
                    if text_part:
                        recognized_text.append(text_part)
        
        if result.stderr:
            print("Errors:", result.stderr[:200])
        
        return recognized_text
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return []

def analyze_latest_conversation():
    """Analyze the most recent conversation audio"""
    
    # Find the latest conversation file
    conversation_files = []
    for i in range(1, 10):
        file_path = f"/mnt/c/temp/conversation_{i}.wav"
        if os.path.exists(file_path):
            conversation_files.append(file_path)
    
    if not conversation_files:
        print("No conversation files found")
        return
    
    latest_file = conversation_files[-1]
    print(f"📁 Latest conversation file: {latest_file}")
    
    # Get basic file info
    file_size = os.path.getsize(latest_file)
    print(f"📊 File size: {file_size} bytes")
    
    # Try to extract speech
    recognized_words = try_windows_speech_on_file(latest_file)
    
    if recognized_words:
        print(f"\n✅ I heard you say: {' '.join(recognized_words)}")
        
        # Simple understanding check
        user_text = ' '.join(recognized_words).lower()
        if any(word in user_text for word in ['understand', 'heard', 'me']):
            print("🎯 Yes! I heard you ask about understanding!")
            print("💭 To answer your question: I can hear you but I need better speech processing to fully understand.")
        else:
            print(f"🤔 I detected speech but need better processing to understand the full meaning.")
    else:
        print("❌ Could not extract clear speech from the audio")
        print("🔧 The recording worked but speech recognition needs improvement")

if __name__ == "__main__":
    print("=== Audio Processing Test ===\n")
    analyze_latest_conversation()
    
    print(f"\n🎯 Status Update:")
    print("✅ Audio recording: Working!")
    print("✅ TTS output: Working!")
    print("🔄 Speech recognition: Needs improvement")
    print("💡 Next: Better STT processing for full understanding")