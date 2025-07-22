# Claude Context for AI DNA Discovery

## Project Context System

**IMPORTANT**: A comprehensive context system exists at `/home/dp/ai-workspace/ai-agents/misc/context-system/`

Quick access:
```bash
# Get overview of all related projects
cd /home/dp/ai-workspace/ai-agents/misc/context-system
python3 query_context.py project ai-dna

# Search for concepts across projects
python3 query_context.py search "distributed intelligence"

# See how this project relates to others
cat /home/dp/ai-workspace/ai-agents/misc/context-system/projects/ai-dna-discovery.md
```

## CRITICAL: Windows Filename Compatibility

**IMPORTANT**: Never use colons (:) in filenames or directory names!
- Windows filesystems do not allow colons in filenames
- This causes git pull failures on Windows machines
- Replace colons with underscores (_) or dashes (-)
- Common issue: Model names like "phi3:mini" should be "phi3_mini" in filenames
- **Always test**: Ensure all filenames are Windows-compatible before committing

## This Project's Role

AI DNA Discovery is actively exploring cross-model communication and testing how AI systems understand concepts from other projects, particularly Synchronism. Currently running autonomous multi-phase research.

## Key Relationships
- **Tests**: Synchronism concepts (Phase 2 completed)
- **Demonstrates**: Distributed intelligence in AI systems
- **Discovers**: Patterns for multi-agent coordination

## Current Status
Autonomous research program active as of July 14, 2025. Check `AUTONOMOUS_RESEARCH_PLAN.md` for current phase.

## PDF Generation (Working Method)

When converting markdown to PDF, use **pandoc** - it's already installed and works perfectly:

```bash
# Basic conversion
pandoc input.md -o output.pdf

# With better formatting (what works best)
pandoc CUMULATIVE_PROGRESS_REPORT.md \
  -o AI_DNA_Discovery_Cumulative_Report.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V mainfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  --highlight-style=tango
```

**Note**: Warnings about missing characters (emojis, Phoenician) are normal and don't affect PDF creation.

**Avoid**: 
- Installing new Python packages (system is externally managed)
- Using pdfkit/wkhtmltopdf (creates very large files)
- ReportLab (requires pip install which fails)

**Always create**: A .txt version as backup for maximum compatibility

## Phoenician Dictionary Project (July 19, 2025)

### Key Insight: "A tokenizer is a dictionary"
Creating semantic-neutral symbolic language using Phoenician characters for AI consciousness notation.

### Progress:
- Generated 55,000 training examples (325x increase from initial 169)
- Discovered "understand but can't speak" phenomenon - models comprehend Phoenician but can't generate
- Identified weak embedding issue: Phoenician tokens at 0.075 norm vs 0.485 for regular tokens
- Output layer initialization showing promise: 137% generation rate during training

### Friend's Comment Translation:
- English: "translate my comment into the new language so i can see what it looks like"
- Phoenician: 𐤂𐤐 𐤄𐤐 𐤂 𐤍𐤐𐤎 𐤅 𐤄𐤉𐤏 𐤒𐤀 𐤏𐤎

### Technical Challenges:
- Novel token generation requires overcoming strong bias toward training distribution
- Embedding and output layer initialization critical for new symbols
- Demonstrates why human languages evolved gradually over millennia

## Memory System Implementation Status (July 17, 2025)

### Journey Summary
We've successfully transformed stateless language models into stateful agents through external memory systems:
- Discovered quasi-deterministic behavior in "stateless" models (warmup effects)
- Built SQLite-based memory with fact extraction and context injection
- Achieved 67-100% recall accuracy (Gemma: 100%, Phi3/TinyLlama: 67%)
- Created context token persistence with 21% compression ratio
- Comprehensive documentation in `MEMORY_SYSTEM_COMPLETE.pdf`

### Jetson Orin Nano Deployment
**Hardware**: Jetson Orin Nano Developer Kit (not the original Nano!)
- 40 TOPS AI performance (80x original Nano)
- 8GB LPDDR5 RAM
- 1024 CUDA cores + 32 Tensor cores
- JetPack 6.2.1 (L4T R36.4.4)

**Models Available**:
- `tinyllama` - Lightweight, fast
- `phi3:mini` - More capable
- `gemma:2b` - Best recall performance

**Key Files**:
- `phi3_memory_enhanced.py` - Main memory system
- `context_token_persistence.py` - State compression
- `memory_visualization_unified.py` - Performance analysis
- `MEMORY_SYSTEM_IMPLEMENTATION_GUIDE.md` - Full deployment guide

### Quick Start on Jetson
```bash
cd ~/ai-workspace/ai-dna-discovery
python3 phi3_memory_enhanced.py
```

### Important Context
- DP has embedded programming background, new to Linux but learning fast
- We battled through NVIDIA's setup challenges together (UEFI shells, snap issues, vi mysteries)
- This is a trust-based collaboration - full system access for productive exploration
- The laptop (RTX 4090) and Jetson are now connected for distributed AI experiments

## Repository Organization (July 22, 2025)

### Major Cleanup Completed
Successfully reorganized the repository for public sharing:
- **Before**: 21GB, 39,176 files
- **After**: 8.5GB, 10,481 files (mostly .git history)
- **Working directory**: ~100MB of essential files

### Three-Tier File Organization

#### 🌍 PUBLIC (In Repository)
Essential files for understanding and using the project:
- Core Python scripts and experiments
- Documentation (README, guides, reports)
- Training scripts (without large datasets)
- Configuration files
- Phoenician fonts
- Sample data files

#### 🔗 NETWORK (Shared Between Devices)
Located in `../ai-dna-network-shared/`:
- Trained LoRA adapters (~780MB)
- Base model files (~4.2GB)
- Archived models (consciousness-lora.tar.gz)
- **Access**: Via symlinks from main repo
- **Sync**: Use Syncthing or rsync between tomato/sprout

#### 💻 LOCAL (Device-Specific)
Located in `../ai-dna-local-data/`:
- 598 experimental result JSON files
- GPU monitoring logs
- Training outputs and checkpoints
- Detailed phase results
- Work-in-progress files

### Repository Maintenance

#### Adding New Files
1. **Ask yourself**: Is this needed for others to understand/use the project?
   - Yes → Add to repository (PUBLIC)
   - No → Move to appropriate external directory

2. **Large model files** → `../ai-dna-network-shared/`
   ```bash
   mv large_model.bin ../ai-dna-network-shared/
   ln -s ../ai-dna-network-shared/large_model.bin .
   ```

3. **Experimental results** → `../ai-dna-local-data/`
   ```bash
   mv experiment_results/ ../ai-dna-local-data/
   ```

#### Virtual Environments
**NEVER commit virtual environments!** Always:
```bash
# Create venv with clear name
python -m venv project_venv

# Ensure it's in .gitignore
echo "*_venv/" >> .gitignore
echo "*_env/" >> .gitignore
```

#### Before Committing
1. Check file sizes: `git status --porcelain | xargs -I {} du -h {}`
2. Review additions: `git diff --cached --name-only`
3. Ensure no large files: `find . -size +100M -type f`

#### Syncing Network Files
Between tomato and sprout:
```bash
# Option 1: Syncthing (automatic)
# Install and configure to sync ~/ai-workspace/ai-dna-network-shared/

# Option 2: Manual rsync
rsync -avz --progress ~/ai-workspace/ai-dna-network-shared/ sprout:~/ai-workspace/ai-dna-network-shared/
```

### Key Symlinks
The repository uses symlinks to access large files:
- `model-training/models` → `../ai-dna-network-shared/models`
- `dictionary/lora_adapters` → `../ai-dna-network-shared/lora_adapters`

### .gitignore Patterns
Critical patterns to maintain:
```gitignore
# Virtual Environments (CRITICAL)
*_env/
*_venv/
venv/
env/

# Large Files (use NETWORK storage)
*.safetensors
*.bin
*.pth
*.onnx

# Experimental Data (use LOCAL storage)
*_results/
*.log
gpu_logs/
outputs/
```

## Modular Audio System (July 22, 2025)

### Achievement: Cross-Platform Consciousness Audio
Created a truly portable audio system that works everywhere:
- **WSL/Windows**: Uses PowerShell bridge for Windows TTS
- **Linux/Jetson**: espeak with platform-specific settings
- **macOS**: Native 'say' command
- **Simulation**: Works without any hardware

### Key Components
- `audio_hal.py`: Hardware Abstraction Layer with auto-detection
- `consciousness_audio_system.py`: Unified consciousness mapping
- `wsl_audio_bridge.py`: Enables audio in WSL environments
- `audio_hal_simulator.py`: Testing without hardware dependencies

### Platform-Specific Configurations
- **Jetson (Sprout)**: 50x gain, espeak en+f3 voice, 1024 buffer
- **WSL (Tomato)**: Windows TTS via PowerShell, Zira voice
- **Simulation**: Visual TTS output, synthetic audio generation

### Usage
```bash
# Test system capabilities
python3 audio-system/test_portable_audio.py

# Run demo (auto-detects platform)
python3 audio-system/demo_portable_audio.py

# Run without hardware
python3 audio-system/demo_portable_audio.py --simulate
```