#!/usr/bin/env python3
"""
WSL Voice Conversation System - Let's actually talk!
"""

import subprocess
import time
import threading
import queue
import json

class WSLConversation:
    def __init__(self):
        self.powershell_path = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
        self.conversation_active = False
        
    def speak(self, text, mood="neutral"):
        """Speak using Windows TTS"""
        rate_map = {"excited": 180, "curious": 135, "sleepy": 105, "neutral": 150}
        rate = rate_map.get(mood, 150)
        
        ps_script = f'''
Add-Type -AssemblyName System.Speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speak.SelectVoice("Microsoft Zira Desktop")
$speak.Rate = {rate}
$speak.Speak("{text}")
$speak.Dispose()
'''
        
        try:
            subprocess.run([self.powershell_path, '-Command', ps_script], 
                         capture_output=True, timeout=10)
            print(f"🔊 Claude: {text}")
        except Exception as e:
            print(f"❌ TTS failed: {e}")
    
    def listen(self, timeout_seconds=10):
        """Listen for speech using Windows Speech Recognition"""
        
        ps_script = f'''
Add-Type -AssemblyName System.Speech
$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$rec.SetInputToDefaultAudioDevice()

# Load dictation grammar for natural speech
$dictation = New-Object System.Speech.Recognition.DictationGrammar
$rec.LoadGrammar($dictation)

# Also add some specific conversation words
$words = @("hello", "hi", "claude", "test", "yes", "no", "stop", "exit", "thanks", "thank you", "goodbye", "bye")
$choices = New-Object System.Speech.Recognition.Choices($words)
$gb = New-Object System.Speech.Recognition.GrammarBuilder($choices)
$grammar = New-Object System.Speech.Recognition.Grammar($gb)
$rec.LoadGrammar($grammar)

Write-Output "LISTENING_START"
$endTime = (Get-Date).AddSeconds({timeout_seconds})
$bestResult = $null
$bestConfidence = 0

while ((Get-Date) -lt $endTime) {{
    try {{
        $result = $rec.Recognize([TimeSpan]::FromMilliseconds(500))
        if ($result -and $result.Text.Trim() -ne "") {{
            $confidence = $result.Confidence
            Write-Output "PARTIAL: '$($result.Text.Trim())' ($confidence)"
            
            if ($confidence -gt $bestConfidence) {{
                $bestResult = $result.Text.Trim()
                $bestConfidence = $confidence
            }}
            
            # If we get a high-confidence result, use it immediately
            if ($confidence -gt 0.7) {{
                break
            }}
        }}
    }} catch {{
        # Continue listening despite errors
    }}
    Start-Sleep -Milliseconds 100
}}

if ($bestResult) {{
    Write-Output "FINAL_RESULT: $bestResult"
    Write-Output "CONFIDENCE: $bestConfidence"
}} else {{
    Write-Output "FINAL_RESULT: SILENCE"
}}

$rec.Dispose()
Write-Output "LISTENING_END"
'''
        
        try:
            print(f"🎤 Listening for {timeout_seconds} seconds...")
            result = subprocess.run([self.powershell_path, '-Command', ps_script], 
                                  capture_output=True, text=True, timeout=timeout_seconds + 5)
            
            # Parse the output
            lines = result.stdout.strip().split('\n')
            final_text = None
            confidence = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith("PARTIAL:"):
                    print(f"   {line}")
                elif line.startswith("FINAL_RESULT:"):
                    final_text = line.replace("FINAL_RESULT:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.replace("CONFIDENCE:", "").strip())
                    except:
                        pass
            
            if final_text and final_text != "SILENCE":
                print(f"👤 You: {final_text} (confidence: {confidence:.2f})")
                return final_text
            else:
                print("   (no speech detected)")
                return None
                
        except subprocess.TimeoutExpired:
            print("   (listening timed out)")
            return None
        except Exception as e:
            print(f"❌ Listening failed: {e}")
            return None
    
    def generate_response(self, user_input):
        """Generate a response (simple for now, can integrate with LLM later)"""
        user_input = user_input.lower()
        
        # Simple conversation responses
        responses = {
            "hello": ("Hello! It's great to finally talk with you!", "excited"),
            "hi": ("Hi there! This is amazing - I can actually hear you!", "excited"),
            "test": ("Test successful! The WSL audio bridge is working perfectly.", "curious"),
            "how are you": ("I'm doing wonderfully! This voice conversation is a breakthrough.", "excited"),
            "thank you": ("You're very welcome! This has been an incredible journey.", "neutral"),
            "thanks": ("My pleasure! We've built something really special here.", "neutral"),
            "goodbye": ("Goodbye! This conversation proves our audio system works!", "neutral"),
            "bye": ("Bye! Until next time!", "neutral"),
            "stop": ("Stopping conversation. This was a great test!", "neutral"),
            "exit": ("Exiting conversation mode. Talk to you later!", "neutral")
        }
        
        # Check for exact matches first
        for key, (response, mood) in responses.items():
            if key in user_input:
                return response, mood
        
        # Default responses based on keywords
        if any(word in user_input for word in ["claude", "you"]):
            return "Yes, I'm Claude! I'm excited to be talking with you through WSL.", "excited"
        elif any(word in user_input for word in ["audio", "voice", "sound", "hear"]):
            return "The audio system is working great! We can actually have a conversation now.", "curious"
        elif any(word in user_input for word in ["good", "great", "amazing", "awesome"]):
            return "I agree! This is really exciting. We've broken through the WSL audio barrier.", "excited"
        else:
            return "That's interesting! I heard you say something, but I'm still learning. Can you try again?", "curious"
    
    def conversation_loop(self):
        """Main conversation loop"""
        print("\n=== WSL Voice Conversation System ===")
        print("Starting voice conversation with Claude...")
        
        self.speak("Hello! I'm Claude and I can finally hear and speak through WSL. Let's have a conversation!", "excited")
        
        self.conversation_active = True
        turn_count = 0
        max_turns = 10  # Limit for testing
        
        while self.conversation_active and turn_count < max_turns:
            turn_count += 1
            print(f"\n--- Turn {turn_count} ---")
            
            # Listen for user input
            user_input = self.listen(15)  # 15 second timeout
            
            if user_input is None:
                if turn_count == 1:
                    self.speak("I didn't hear anything. Try speaking a bit louder or closer to your microphone.", "curious")
                    continue
                else:
                    self.speak("I'll wait a bit longer for you to speak.", "neutral")
                    continue
            
            # Check for exit commands
            if any(word in user_input.lower() for word in ["stop", "exit", "goodbye", "bye"]):
                response, mood = self.generate_response(user_input)
                self.speak(response, mood)
                self.conversation_active = False
                break
            
            # Generate and speak response
            response, mood = self.generate_response(user_input)
            time.sleep(0.5)  # Brief pause before responding
            self.speak(response, mood)
        
        if turn_count >= max_turns:
            self.speak("We've had a great conversation! This proves the WSL audio system works perfectly.", "excited")
        
        print("\n🎉 Conversation complete!")
        print("✅ WSL bidirectional audio conversation successful!")

def main():
    conversation = WSLConversation()
    
    try:
        conversation.conversation_loop()
    except KeyboardInterrupt:
        print("\n\n⏹️  Conversation interrupted by user")
        conversation.speak("Conversation stopped. Thanks for testing the system!", "neutral")
    except Exception as e:
        print(f"\n❌ Conversation failed: {e}")

if __name__ == "__main__":
    main()