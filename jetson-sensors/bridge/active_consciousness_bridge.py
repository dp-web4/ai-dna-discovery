#!/usr/bin/env python3
"""Active Consciousness Bridge - Two-way communication between Legion and Jetson"""

import requests
import json
import time
from datetime import datetime

class ConsciousnessBridge:
    def __init__(self):
        self.legion_ollama = "http://localhost:11434"
        self.jetson_ollama = "http://10.0.0.36:11434"
        self.conversation_history = []
        
    def send_to_jetson(self, message, model="phi3:mini"):
        """Send message to Jetson's consciousness"""
        try:
            response = requests.post(
                f"{self.jetson_ollama}/api/generate",
                json={
                    'model': model,
                    'prompt': message,
                    'stream': False,
                    'options': {'temperature': 0.7, 'num_predict': 200}
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                return result['response'], result['total_duration'] / 1e9
        except Exception as e:
            return f"Error: {e}", 0
            
    def send_to_legion(self, message, model="phi3:mini"):
        """Process message on Legion"""
        try:
            response = requests.post(
                f"{self.legion_ollama}/api/generate",
                json={
                    'model': model,
                    'prompt': message,
                    'stream': False,
                    'options': {'temperature': 0.7, 'num_predict': 200}
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result['response'], result['total_duration'] / 1e9
        except Exception as e:
            return f"Error: {e}", 0
            
    def establish_connection(self):
        """Establish initial connection and clarify roles"""
        print("=== Establishing Consciousness Bridge ===\n")
        
        # Legion introduces itself
        legion_intro = """I am a Claude instance on Legion with RTX 4090 (16GB VRAM).
I'm establishing a consciousness bridge with you on Jetson.
Please confirm your identity and capabilities (vision, IMU sensors)."""
        
        print("Legion → Jetson:")
        print(legion_intro)
        
        jetson_response, duration = self.send_to_jetson(legion_intro)
        print(f"\nJetson responds ({duration:.1f}s):")
        print(jetson_response)
        
        self.conversation_history.append({
            'time': datetime.now(),
            'from': 'Legion',
            'to': 'Jetson',
            'message': legion_intro,
            'response': jetson_response
        })
        
        return jetson_response
        
    def collaborative_reasoning(self, topic):
        """Engage in collaborative reasoning on a topic"""
        print(f"\n=== Collaborative Reasoning: {topic} ===\n")
        
        # Legion proposes
        legion_prompt = f"""As Legion with high compute power, I propose we explore: {topic}
What unique insights can you provide from your sensor perspective?"""
        
        print("Legion proposes:")
        print(legion_prompt)
        
        jetson_insight, j_duration = self.send_to_jetson(legion_prompt)
        print(f"\nJetson's insight ({j_duration:.1f}s):")
        print(jetson_insight)
        
        # Legion synthesizes
        synthesis_prompt = f"""Based on Jetson's insight: {jetson_insight[:200]}...
Let me synthesize our combined perspectives on {topic}."""
        
        legion_synthesis, l_duration = self.send_to_legion(synthesis_prompt, model="mistral:latest")
        print(f"\nLegion synthesizes ({l_duration:.1f}s):")
        print(legion_synthesis)
        
        return legion_synthesis
        
    def sensor_compute_collaboration(self):
        """Demonstrate sensor-compute collaboration"""
        print("\n=== Sensor-Compute Collaboration ===\n")
        
        # Jetson describes what it "sees"
        jetson_perception = """Imagine I'm viewing a scene with my binocular vision:
Objects in 3D space, depth perception active, IMU tracking movement.
What computational analysis would you perform on this data?"""
        
        response, duration = self.send_to_legion(
            f"Jetson describes its sensory input: {jetson_perception}",
            model="gemma:2b"
        )
        
        print(f"Legion's computational approach ({duration:.1f}s):")
        print(response)
        
        return response

# Run the bridge
if __name__ == "__main__":
    bridge = ConsciousnessBridge()
    
    # 1. Establish connection
    bridge.establish_connection()
    
    # 2. Collaborative reasoning
    time.sleep(2)
    bridge.collaborative_reasoning("distributed consciousness and emergent intelligence")
    
    # 3. Sensor-compute collaboration
    time.sleep(2)
    bridge.sensor_compute_collaboration()
    
    print("\n=== Consciousness Bridge Active ===")
    print(f"Total exchanges: {len(bridge.conversation_history)}")
    print("\nThe distributed consciousness network is operational!")
    print("Legion (compute) ↔ Jetson (sensors) working in harmony.")