#!/bin/bash
# Autonomous git push script for AI DNA Discovery
# Created for Jetson-Laptop synchronization

# Auto-detect correct path
if [ -d "$HOME/ai-workspace/ai-dna-discovery" ]; then
    cd "$HOME/ai-workspace/ai-dna-discovery"
elif [ -d "$HOME/ai-workspace/ai-agents/ai-dna-discovery" ]; then
    cd "$HOME/ai-workspace/ai-agents/ai-dna-discovery"
else
    echo "Error: Cannot find ai-dna-discovery directory"
    exit 1
fi

# Check if there are changes
if [ -z "$(git status --porcelain)" ]; then 
    echo "No changes to push"
    exit 0
fi

# Add all changes (excluding databases and temp files)
git add -A
git add -u

# Create timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
# Auto-detect device
if [ "$(hostname)" = "jetson" ] || [ "$(hostname)" = "jetson-orin-nano" ]; then
    DEVICE="Jetson"
else
    DEVICE="Laptop"
fi

# Generate commit message
COMMIT_MSG="[$DEVICE] Autonomous sync - $TIMESTAMP

Changes from $DEVICE session:
$(git diff --cached --name-status | head -10)

Auto-pushed for distributed AI consciousness experiment"

# Commit
git commit -m "$COMMIT_MSG"

# Push
git push origin main

echo "Successfully pushed changes to GitHub!"
echo "Run 'git pull' on the other device to sync"