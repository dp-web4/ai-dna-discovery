#!/bin/bash
# Autonomous git push script for AI DNA Discovery
# Created for Jetson-Laptop synchronization

cd ~/ai-workspace/ai-dna-discovery

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
DEVICE="Jetson"  # Change to "Laptop" when running there

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