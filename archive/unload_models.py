#!/usr/bin/env python3
"""Unload all models except the one we're testing"""

import ollama
import requests

# Get currently loaded models
response = requests.get("http://localhost:11434/api/ps")
loaded = response.json()

print("Currently loaded models:")
for model in loaded.get('models', []):
    print(f"  - {model['name']}")

# Unload all except deepseek-coder
print("\nUnloading models...")
for model in loaded.get('models', []):
    if model['name'] != 'deepseek-coder:1.3b':
        print(f"  Unloading {model['name']}...")
        # Generate with keep_alive=0 to unload
        ollama.generate(
            model=model['name'],
            prompt="",
            keep_alive="0"
        )

print("\nDone. Only deepseek-coder should remain.")