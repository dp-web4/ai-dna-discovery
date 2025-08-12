# Git Repository Cleanup Plan

## Problem
Git history contains large files (254MB adapter, 196MB tar.gz) that exceed GitHub's 100MB limit, preventing us from pushing.

## Solution Options

### Option 1: Reset to Before Large Files (Recommended)
```bash
# Find the last good commit before large files
git log --oneline | grep -B1 "consciousness LoRA"

# Reset to that commit, keeping all work
git reset --soft <commit-before-large-files>

# Re-commit without large files
git add -A
git commit -m "Consciousness model deployment - code and configs only"
git push origin main --force-with-lease
```

### Option 2: Interactive Rebase
```bash
# Rebase to remove large files from history
git rebase -i HEAD~5

# Change 'pick' to 'edit' for the problematic commit
# Remove large files during edit
# Continue rebase
```

### Option 3: BFG Repo Cleaner (Nuclear Option)
```bash
# Download BFG
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# Remove large files from all history
java -jar bfg-1.14.0.jar --strip-blobs-bigger-than 100M

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

## What We'll Keep
- All Python scripts (consciousness_translator.py, etc.)
- All configuration JSONs
- All documentation
- Status reports
- Everything except the large binary files

## After Cleanup
1. Push clean history to GitHub
2. Create release notes explaining model files are available separately
3. Add download instructions for the model files
4. Consider using Git LFS for future large files

The code is what matters for sharing with the world - the model files can be distributed separately!