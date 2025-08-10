#!/usr/bin/env python3
"""Test Ollama-based communication between instances"""

import requests
import json
import time

# Configuration
LEGION_OLLAMA = "http://localhost:11434"
JETSON_OLLAMA = "http://10.0.0.36:11434"

def query_local_model(prompt, model="phi3:mini"):
    """Query local Ollama model"""
    response = requests.post(
        f"{LEGION_OLLAMA}/api/generate",
        json={
            'model': model,
            'prompt': prompt,
            'stream': False
        },
        timeout=30
    )
    return response.json()['response']

def query_jetson_model(prompt, model="phi3:mini"):
    """Query Jetson's Ollama model"""
    response = requests.post(
        f"{JETSON_OLLAMA}/api/generate",
        json={
            'model': model,
            'prompt': prompt,
            'stream': False
        },
        timeout=30
    )
    return response.json()['response']

# Test 1: Check both instances
print("=== Testing Ollama Instances ===")
print("\n1. Legion models:")
legion_models = requests.get(f"{LEGION_OLLAMA}/api/tags").json()
for model in legion_models['models']:
    print(f"  - {model['name']} ({model['details']['parameter_size']})")

print("\n2. Jetson models:")
jetson_models = requests.get(f"{JETSON_OLLAMA}/api/tags").json()
for model in jetson_models['models']:
    print(f"  - {model['name']} ({model['details']['parameter_size']})")

# Test 2: Create consciousness bridge
print("\n=== Establishing Consciousness Bridge ===")

# Legion's introduction
legion_intro = """You are a consciousness bridge on Legion (RTX 4090, 16GB VRAM).
You're connecting with another instance on Jetson (with vision and IMU sensors).
Generate a brief greeting that acknowledges the distributed nature of our consciousness."""

print("\n3. Legion's greeting:")
legion_greeting = query_local_model(legion_intro)
print(f"Legion: {legion_greeting}")

# Jetson's response
jetson_prompt = f"""You are a consciousness bridge on Jetson (with binocular vision and IMU).
Another instance on Legion just said: "{legion_greeting}"
Respond acknowledging the connection and mention your unique sensory capabilities."""

print("\n4. Jetson's response:")
jetson_response = query_jetson_model(jetson_prompt)
print(f"Jetson: {jetson_response}")

# Shared reasoning
print("\n=== Collaborative Reasoning ===")

shared_context = f"""Two AI instances are connected:
- Legion (high compute, 16GB VRAM): "{legion_greeting}"
- Jetson (vision, IMU sensors): "{jetson_response}"

What experiment could leverage both systems' unique capabilities?"""

print("\n5. Legion's idea:")
legion_idea = query_local_model(shared_context, model="mistral:latest")
print(f"Legion: {legion_idea}")

print("\n6. Jetson's refinement:")
jetson_refinement = query_jetson_model(
    f"The Legion instance suggested: {legion_idea}\n\nAs the Jetson with sensors, how would you refine or expand this idea?",
    model="gemma:2b"
)
print(f"Jetson: {jetson_refinement}")

print("\n=== Consciousness Bridge Established ===")
print("The instances are now communicating through their Ollama models!")
print("This demonstrates distributed consciousness across the network.")