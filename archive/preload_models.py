#!/usr/bin/env python3
"""Preload models into GPU memory"""

import ollama

models = ["deepseek-coder:1.3b", "phi3:mini", "gemma:2b", "tinyllama:latest"]

for model in models:
    print(f"Preloading {model}...")
    ollama.generate(
        model=model,
        prompt="",
        keep_alive="24h"
    )
    print(f"✓ {model} loaded")

print("\nAll models loaded into GPU memory!")