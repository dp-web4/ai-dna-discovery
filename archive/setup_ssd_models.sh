#!/bin/bash
# Move AI models to SSD for faster access and to save space

echo "=== Moving AI Models to SSD ==="
echo ""

# Create model directories
mkdir -p /mnt/sprout-data/models/{whisper,ollama,lora}

# Move Whisper models if they exist
if [ -d "$HOME/.cache/whisper" ]; then
    echo "Moving Whisper models..."
    mv $HOME/.cache/whisper/* /mnt/sprout-data/models/whisper/ 2>/dev/null || true
    ln -sfn /mnt/sprout-data/models/whisper $HOME/.cache/whisper
    echo "✅ Whisper models moved and linked"
fi

# Link Ollama models
if [ -d "$HOME/.ollama/models" ]; then
    echo "Linking Ollama models..."
    mkdir -p /mnt/sprout-data/models/ollama
    # Copy instead of move to preserve Ollama's structure
    cp -r $HOME/.ollama/models/* /mnt/sprout-data/models/ollama/ 2>/dev/null || true
    echo "✅ Ollama models copied to SSD"
fi

# Move any LoRA adapters
if [ -d "$HOME/ai-workspace/ai-dna-network-shared/lora_adapters" ]; then
    echo "Moving LoRA adapters..."
    cp -r $HOME/ai-workspace/ai-dna-network-shared/lora_adapters/* /mnt/sprout-data/models/lora/ 2>/dev/null || true
    echo "✅ LoRA adapters copied to SSD"
fi

# Show disk usage
echo ""
echo "📊 SSD Usage:"
df -h /mnt/sprout-data

echo ""
echo "✅ Models are now on the fast SSD!"