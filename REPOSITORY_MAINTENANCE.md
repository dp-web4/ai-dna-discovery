# AI DNA Discovery - Repository Maintenance Guide

## Overview

This repository has been organized into a three-tier system to balance public sharing, device synchronization, and local storage needs. This guide ensures the repository remains clean and manageable.

## Current Structure

### Repository Stats (July 22, 2025)
- **Public Repo Size**: ~100MB (excluding .git)
- **File Count**: ~10,000 files
- **External Storage**: ~6.5GB across two directories

### Directory Layout
```
ai-workspace/
├── ai-dna-discovery/          # Main repository (PUBLIC)
├── ai-dna-local-data/         # Device-specific data (LOCAL)
└── ai-dna-network-shared/     # Shared between devices (NETWORK)
```

## File Classification Guide

### 🌍 PUBLIC - Goes in Repository
✅ Include:
- Python scripts demonstrating methods
- Documentation (README, guides, reports)
- Configuration files
- Small sample datasets (<1MB)
- Essential fonts and resources
- Final PDF reports

❌ Exclude:
- Virtual environments
- Large model files (>10MB)
- Training outputs
- Experimental results
- GPU logs
- Personal notes

### 🔗 NETWORK - Share Between Devices
Store in `../ai-dna-network-shared/`:
- Trained LoRA adapters
- Base model files
- Large datasets for training
- Model checkpoints
- Archived models

Access via symlinks:
```bash
ln -s ../ai-dna-network-shared/models model-training/models
```

### 💻 LOCAL - Keep on Device
Store in `../ai-dna-local-data/`:
- Experimental results (JSON files)
- GPU monitoring logs
- Training outputs
- Debug logs
- Work-in-progress files
- Temporary data

## Daily Workflow

### Before Starting Work
```bash
# Check repository size
du -sh .
find . -type f | wc -l

# Pull latest changes
git pull

# Check symlinks are valid
ls -la model-training/models
ls -la dictionary/lora_adapters
```

### Adding New Files

1. **Small script or doc?** → Add normally
   ```bash
   git add new_experiment.py
   ```

2. **Large model file?** → Network storage
   ```bash
   mv trained_model.bin ../ai-dna-network-shared/
   ln -s ../ai-dna-network-shared/trained_model.bin .
   git add trained_model.bin  # Git tracks the symlink
   ```

3. **Experimental results?** → Local storage
   ```bash
   mv experiment_results/ ../ai-dna-local-data/
   # Don't add to git
   ```

### Before Committing

Always check:
```bash
# Find large files
find . -size +10M -type f | grep -v ".git"

# Check what you're adding
git status
git diff --cached --name-only

# Verify no venv files
git status | grep -E "(_env|_venv|venv/|env/)"
```

## Virtual Environment Management

### Creating Virtual Environments
```bash
# Use descriptive names
python -m venv consciousness_venv
python -m venv phoenician_env

# NEVER use generic names like 'env' or 'venv'
```

### Ensure .gitignore Coverage
Our .gitignore includes:
```
*_env/
*_venv/
venv/
env/
.venv/
```

### If You Accidentally Commit a Venv
```bash
# Remove from tracking
git rm -r --cached accidentally_added_env/
echo "accidentally_added_env/" >> .gitignore
git add .gitignore
git commit -m "Remove virtual environment from tracking"
```

## Syncing Between Devices

### Network Files (tomato ↔ sprout)

#### Option 1: Syncthing (Recommended)
1. Install on both devices
2. Share `~/ai-workspace/ai-dna-network-shared/`
3. Files sync automatically when online

#### Option 2: Manual rsync
```bash
# From tomato to sprout
rsync -avz --progress \
  ~/ai-workspace/ai-dna-network-shared/ \
  sprout:~/ai-workspace/ai-dna-network-shared/

# From sprout to tomato  
rsync -avz --progress \
  ~/ai-workspace/ai-dna-network-shared/ \
  tomato:~/ai-workspace/ai-dna-network-shared/
```

#### Option 3: Quick network transfer
```bash
# On receiving device
cd ~/ai-workspace
nc -l 1234 | tar xzf -

# On sending device
cd ~/ai-workspace
tar czf - ai-dna-network-shared | nc other-device 1234
```

## Common Issues and Solutions

### "File too large" error
```bash
# Find the culprit
find . -size +100M -type f

# Move to appropriate storage
mv large_file.bin ../ai-dna-network-shared/
ln -s ../ai-dna-network-shared/large_file.bin .
```

### Symlink broken after clone
```bash
# Recreate network directory structure
mkdir -p ../ai-dna-network-shared/models
mkdir -p ../ai-dna-network-shared/lora_adapters

# Populate from backup or other device
rsync -avz tomato:~/ai-workspace/ai-dna-network-shared/ ../ai-dna-network-shared/
```

### Repository growing too large
```bash
# Check git history size
du -sh .git

# If .git is too large, consider:
# 1. Create fresh clone
# 2. Use git-filter-repo to remove large file history
# 3. Start new repository with clean history
```

## Monitoring Repository Health

### Weekly Checks
```bash
# Size check
echo "Repository size: $(du -sh . | cut -f1)"
echo "File count: $(find . -type f | wc -l)"
echo "Git history: $(du -sh .git | cut -f1)"

# Large file check
echo "Files over 10MB:"
find . -size +10M -type f | grep -v ".git"

# Virtual environment check
echo "Checking for venvs:"
find . -type d -name "*env" | grep -v ".git"
```

### Monthly Cleanup
```bash
# Clean Python cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# Remove empty directories
find . -type d -empty -delete

# Git cleanup
git gc --aggressive --prune=now
```

## Emergency Recovery

### If you accidentally commit large files
```bash
# Soft reset (keeps changes)
git reset --soft HEAD~1

# Move large files out
mv large_files ../ai-dna-network-shared/

# Recommit without large files
git add .
git commit -m "Previous commit without large files"
```

### If repository becomes corrupted
1. Clone fresh from GitHub
2. Restore network files from other device
3. Local files should be intact in `../ai-dna-local-data/`

## Best Practices

1. **Think before adding**: Does this need to be in the repo?
2. **Use descriptive names**: Make virtual environments obvious
3. **Check file sizes**: Nothing over 10MB without careful consideration
4. **Maintain symlinks**: Keep them pointing to valid locations
5. **Regular cleanup**: Run monthly maintenance
6. **Document changes**: Update this guide when workflow changes

## Quick Reference Card

```bash
# Add model to network storage
mv model.bin ../ai-dna-network-shared/ && ln -s ../ai-dna-network-shared/model.bin .

# Add results to local storage  
mv results/ ../ai-dna-local-data/

# Check before commit
find . -size +10M -type f | grep -v ".git"

# Sync with other device
rsync -avz ../ai-dna-network-shared/ other-device:~/ai-workspace/ai-dna-network-shared/

# Create venv safely
python -m venv project_name_venv
```

Remember: When in doubt, keep it out (of the repository)!