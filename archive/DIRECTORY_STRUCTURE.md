# AI DNA Discovery - Directory Structure

**Total Local Size**: ~21GB  
**Total File Count**: ~39,176 files  
**Last Updated**: July 20, 2025

## Overview

This document maps the directory structure with three classification levels:
- **PUBLIC**: Must be in the repository for others to use
- **NETWORK**: Should be shared between devices (tomato/sprout) but not in public repo
- **LOCAL**: Can stay on individual devices, not needed elsewhere

## Classification Legend
- 🌍 **PUBLIC** - Essential for repository users
- 🔗 **NETWORK** - Share between your devices
- 💻 **LOCAL** - Device-specific, no need to share

## Major Space Consumers

### Virtual Environments (TO BE REMOVED)
These should be in .gitignore and removed from the repository:

1. **dictionary/phoenician_env/** (~5.3GB, ~12,000 files)
   - Python virtual environment for Phoenician training
   - Contains all pip packages and dependencies
   - **Action**: Remove and add to .gitignore

2. **model-training/unsloth_env/** (~8.1GB, ~8,000 files)
   - Virtual environment for Unsloth training framework
   - Contains CUDA packages and ML libraries
   - **Action**: Remove and add to .gitignore

3. **sensor-consciousness/sensor_venv/** (Size TBD)
   - Virtual environment for sensor integration
   - **Action**: Remove and add to .gitignore

## Core Project Directories

### 1. 🌍 **Root Level Core Files** (~50 files, ~10MB)
- Main documentation (README.md, CLAUDE.md, etc.)
- Configuration files (.gitignore, requirements.txt)
- Key Python scripts for experiments
- **PUBLIC**: Essential for understanding and using the project

### 2. 💻 **ai_dna_results/** (498 files, ~2.2MB)
- JSON files from continuous AI DNA experiments
- Cycle results from autonomous testing
- **LOCAL**: Historical data, too detailed for public repo
- Consider creating a summary for PUBLIC

### 3. 🔗 **dictionary/** (~6.6GB total)
- **dictionary/lora_adapters/** (780MB) - Trained LoRA models
  - tinyllama/phoenician_adapter/ (522MB) - Main Phoenician model
  - tinyllama/phoenician_focused/ (258MB) - Optimized version
  - **NETWORK**: Share between devices for deployment
- **dictionary/outputs/** (257MB) - Generated outputs
  - **LOCAL**: Keep for reference
- **dictionary/documentation/** - Training guides
  - **PUBLIC**: How-to guides for replication

### 4. 🔗 **model-training/** (~5GB total)
- **models/** (4.2GB) - Base model files and checkpoints
  - **NETWORK**: Too large for repo, share via sync
- **gpu_logs/** (352MB) - Detailed GPU monitoring logs
  - **LOCAL**: Device-specific performance data
- **outputs/** (277MB) - Training outputs and checkpoints
  - **LOCAL**: Intermediate results
- **consciousness-lora.tar.gz** (197MB) - Archived LoRA adapter
  - **NETWORK**: Shareable trained model
- Training scripts and datasets
  - **PUBLIC**: Scripts needed for replication
  - **NETWORK**: Datasets for consistent training

### 5. 🌍 **sensor-consciousness/** (New branch content)
- Sensor integration experiments
- Camera, IMU, microphone interfaces
- Edge deployment tools
- **PUBLIC**: Core integration code
- **LOCAL**: Test outputs and logs

### 6. 💻 **Phase Experiment Results**
- **phase_0_results/** through **phase_5_results/**
- **LOCAL**: Detailed experimental data
- Create summaries for **PUBLIC**

### 7. 💻 **Experiment-Specific Results**
- **embedding_space_results/** - Embedding analysis visualizations
- **memory_transfer_results/** - Memory persistence experiments
- **handshake_results/** - Model handshake protocols
- **complete_handshake_results/** - Full convergence matrices
- **LOCAL**: Raw experimental data
- Select key visualizations for **PUBLIC**

### 8. 🌍 **fonts/** (6.2KB)
- Phoenician font files for PDF generation
- Font configuration for LaTeX
- **PUBLIC**: Required for proper rendering

### 9. 🔗 **Reports and Documentation**
- PDF reports (comprehensive, progress, etc.)
- **PUBLIC**: Final reports and guides
- **LOCAL**: Work-in-progress versions

### 10. 🌍 **Visualization and Core Scripts**
- Key Python scripts demonstrating methods
- Essential PNG diagrams
- **PUBLIC**: Core demonstrations and visualizations

## Files to Clean Up

### Temporary/Cache Files
- `__pycache__` directories (throughout)
- `.pyc` files
- `checkpoint_*` files from interrupted training
- Duplicate PDFs with versioning

### Virtual Environments (Critical to Remove)
1. `dictionary/phoenician_env/` - 5.3GB, 22,729 files
2. `model-training/unsloth_env/` - 16MB, 73 files (mostly symlinks)
3. `sensor-consciousness/sensor_venv/` - 461MB, 5,264 files
Total to remove: ~5.8GB, ~28,066 files

### Estimated Size After Cleanup
- Current: ~21GB with ~39,000 files
- After cleanup: ~3-4GB with ~5,000 files
- Reduction: ~80% in size, ~87% in file count

## Summary by Classification

### 🌍 PUBLIC (For Repository) ~50MB
- Core documentation (README, CLAUDE.md, guides)
- Essential Python scripts for experiments
- Training scripts (without datasets)
- Phoenician fonts and LaTeX configs
- Key visualizations and diagrams
- Final PDF reports
- Sample data and minimal examples

### 🔗 NETWORK (Share Between Devices) ~5.5GB
- Trained LoRA adapters (phoenician, consciousness)
- Base model files (TinyLlama)
- Training datasets (consciousness, phoenician)
- Model archives (tar.gz files)

### 💻 LOCAL (Keep on Device) ~15GB
- All experimental results (JSON files)
- GPU monitoring logs
- Intermediate training outputs
- Virtual environments
- Work-in-progress documents
- Detailed phase results

## Recommended .gitignore for PUBLIC Repo

```gitignore
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

# Large Model Files (use NETWORK sharing instead)
model-training/models/
dictionary/lora_adapters/
*.safetensors
*.bin
*.pth
*.onnx

# Experimental Data (keep LOCAL)
ai_dna_results/
phase_*_results/
*_results/
embedding_space_results/
memory_transfer_results/
handshake_results/

# Logs and Outputs
*.log
gpu_logs/
model-training/outputs/
dictionary/outputs/

# Large Archives
*.tar.gz
*.zip

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Work in Progress
*_WIP*
*_draft*
```

## Essential Files for PUBLIC Repository

### Must Include:
```
README.md
CLAUDE.md
requirements.txt
.gitignore

# Core experiment scripts
continuous_ai_dna_experiment.py
train_phoenician_lora.py
test_phoenician_generation.py
memory_persistence_experiment.py
sensor-consciousness/src/*.py

# Documentation
COMPREHENSIVE_REPORT.md
PDF_CONVERSION_GUIDE.md
DIRECTORY_STRUCTURE.md
dictionary/documentation/*.md

# Fonts for Phoenician
fonts/NotoSansPhoenician-Regular.ttf
fonts/fallback.tex

# Key visualizations (select best)
consciousness_lattice_visualization.png
phoenician_breakthrough.png
embedding_space_analysis.png

# Sample data
sample_consciousness_notation.json
sample_phoenician_dictionary.json
```

## Next Steps

1. **Create comprehensive .gitignore** using the patterns above
2. **Move LOCAL files** out of git tracking:
   ```bash
   # Example: Move results to local storage
   mkdir -p ../ai-dna-local-data
   mv ai_dna_results ../ai-dna-local-data/
   mv phase_*_results ../ai-dna-local-data/
   ```

3. **Set up NETWORK sync** for model sharing:
   ```bash
   # Consider using rsync or Syncthing for model files
   # Between tomato and sprout
   ```

4. **Clean and commit**:
   ```bash
   git add .gitignore
   git rm -r --cached ai_dna_results/
   git rm -r --cached phase_*_results/
   git rm -r --cached model-training/models/
   git rm -r --cached dictionary/lora_adapters/
   git commit -m "Restructure repo for public sharing"
   ```

## Network Sync Recommendations

For NETWORK files between tomato/sprout:
- Use Syncthing for automatic bidirectional sync
- Or rsync with scripts for manual sync
- Keep models in `~/ai-models-shared/` on both devices
- Symlink from project directories as needed