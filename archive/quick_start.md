# AI Workspace Quick Start

## What's Running Now

1. **Ollama** - Local AI model server (using your RTX 4090)
   - API endpoint: http://127.0.0.1:11434
   - Model installed: phi3:mini (3.8B parameters, fast)

2. **MCP Servers** (configured, restart Claude Code to activate)
   - Filesystem server: Access to ~/ai-workspace
   - Memory server: Context persistence

## Quick Commands

### Chat with AI
```bash
ollama run phi3:mini
```

### List models
```bash
ollama list
```

### Pull more models
```bash
# Small & fast
ollama pull tinyllama
ollama pull gemma:2b

# Medium
ollama pull mistral
ollama pull llama3.2

# Large (needs more VRAM)
ollama pull llama3.1:70b
```

### Python API example
```python
import requests
import json

response = requests.post('http://localhost:11434/api/generate',
    json={
        'model': 'phi3:mini',
        'prompt': 'Why is the sky blue?',
        'stream': False
    })
print(json.loads(response.text)['response'])
```

## Next Steps

1. Install CUDA toolkit for PyTorch GPU acceleration
2. Set up Jupyter notebooks for experiments
3. Try larger models (Mistral 7B, LLaMA 3.1)
4. Build custom tools using the local AI API
5. Fine-tune models on your own data

## Your patterns.txt insights
I read your patterns document - fascinating perspective on knowledge, perspective, and the limits of consensus. The elephant analogy is powerful for understanding how we need both distance AND open eyes to see the full picture.