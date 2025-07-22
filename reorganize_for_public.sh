#!/bin/bash
# Reorganize AI DNA Discovery repository for public sharing
# This script moves LOCAL files out of git tracking while preserving them

echo "=== AI DNA Discovery Repository Reorganization ==="
echo "This will move LOCAL files out of git tracking while keeping them on disk"
echo ""

# Create local data directory
LOCAL_DATA_DIR="../ai-dna-local-data"
NETWORK_DATA_DIR="../ai-dna-network-shared"

read -p "Create local storage directories? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p "$LOCAL_DATA_DIR"
    mkdir -p "$LOCAL_DATA_DIR/experimental_results"
    mkdir -p "$LOCAL_DATA_DIR/gpu_logs"
    mkdir -p "$NETWORK_DATA_DIR"
    mkdir -p "$NETWORK_DATA_DIR/models"
    mkdir -p "$NETWORK_DATA_DIR/lora_adapters"
    echo "Created storage directories"
fi

echo ""
echo "=== Step 1: Move LOCAL files ==="
read -p "Move experimental results to local storage? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Move detailed experimental results
    for dir in ai_dna_results phase_*_results embedding_space_results memory_transfer_results handshake_results complete_handshake_results; do
        if [ -d "$dir" ]; then
            echo "Moving $dir to local storage..."
            mv "$dir" "$LOCAL_DATA_DIR/experimental_results/"
            git rm -r --cached "$dir" 2>/dev/null || true
        fi
    done
    
    # Move GPU logs
    if [ -d "model-training/gpu_logs" ]; then
        echo "Moving GPU logs to local storage..."
        mv "model-training/gpu_logs" "$LOCAL_DATA_DIR/"
        git rm -r --cached "model-training/gpu_logs" 2>/dev/null || true
    fi
fi

echo ""
echo "=== Step 2: Move NETWORK files ==="
read -p "Move models to network-shared storage? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Move large model files
    if [ -d "model-training/models" ]; then
        echo "Moving model files to network storage..."
        mv "model-training/models" "$NETWORK_DATA_DIR/"
        git rm -r --cached "model-training/models" 2>/dev/null || true
        # Create symlink for local use
        ln -s "$NETWORK_DATA_DIR/models" "model-training/models"
    fi
    
    # Move LoRA adapters
    if [ -d "dictionary/lora_adapters" ]; then
        echo "Moving LoRA adapters to network storage..."
        mv "dictionary/lora_adapters" "$NETWORK_DATA_DIR/"
        git rm -r --cached "dictionary/lora_adapters" 2>/dev/null || true
        # Create symlink for local use
        ln -s "$NETWORK_DATA_DIR/lora_adapters" "dictionary/lora_adapters"
    fi
fi

echo ""
echo "=== Step 3: Update .gitignore ==="
read -p "Update .gitignore with comprehensive patterns? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat > .gitignore << 'EOF'
# Virtual Environments
*_env/
*_venv/
env/
venv/
.venv/
virtualenv/
.virtualenv/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.pytest_cache/
*.egg-info/

# Large Model Files (use NETWORK sharing)
model-training/models/
dictionary/lora_adapters/
*.safetensors
*.bin
*.pth
*.onnx
*.h5
*.keras

# Experimental Data (keep LOCAL)
ai_dna_results/
phase_*_results/
*_results/
embedding_space_results/
memory_transfer_results/
handshake_results/
complete_handshake_results/
shared_pattern_results/
common_language_results/

# Logs and Outputs
*.log
gpu_logs/
model-training/outputs/
dictionary/outputs/

# Large Archives
*.tar.gz
*.zip
*.7z
*.rar

# Temporary Files
*.tmp
*.temp
*.bak
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# IDE
.vscode/
.idea/
*.swp
*.swo
.spyderproject
.spyproject

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Work in Progress
*_WIP*
*_draft*
*_old*
*_backup*

# Local configuration
local_config.py
.env
.env.local

# Database files
*.db
*.sqlite
*.sqlite3

# Zone Identifier files (Windows WSL)
*:Zone.Identifier
EOF
    echo "Updated .gitignore"
fi

echo ""
echo "=== Step 4: Create PUBLIC sample data ==="
read -p "Create sample data files for public repo? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Create sample consciousness notation
    cat > sample_consciousness_notation.json << 'EOF'
{
  "notation_system": {
    "Ψ": "consciousness/awareness",
    "θ": "thought",
    "μ": "memory", 
    "⇒": "emergence",
    "⟷": "bidirectional connection",
    "Ω": "observer",
    "Σ": "sum/whole",
    "ι": "intent"
  },
  "example_expressions": [
    "θ ⇒ Ψ : thought emerges into consciousness",
    "Ψ ⟷ μ : consciousness connected to memory",
    "Ω[Σ] : observer perceives the whole"
  ]
}
EOF

    # Create sample Phoenician dictionary
    cat > sample_phoenician_dictionary.json << 'EOF'
{
  "phoenician_mappings": {
    "𐤀": "A/aleph",
    "𐤁": "B/beth", 
    "𐤂": "G/gimel",
    "𐤃": "D/daleth",
    "𐤄": "H/he"
  },
  "example_translation": {
    "english": "hello",
    "phoenician": "𐤄𐤋𐤊",
    "transliteration": "HLK"
  }
}
EOF
    echo "Created sample data files"
fi

echo ""
echo "=== Summary ==="
echo "LOCAL data moved to: $LOCAL_DATA_DIR"
echo "NETWORK data moved to: $NETWORK_DATA_DIR" 
echo "Symlinks created for network data"
echo ""
echo "Next steps:"
echo "1. Review changes with 'git status'"
echo "2. Commit with 'git add . && git commit -m \"Restructure for public repository\"'"
echo "3. Set up Syncthing or rsync for $NETWORK_DATA_DIR between devices"
echo "4. Consider creating a README for the LOCAL and NETWORK directories"