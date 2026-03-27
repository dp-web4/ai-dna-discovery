#!/bin/bash
# Machine-aware git push script

echo "🔍 Identifying machine..."
MACHINE=$(hostname)
echo "Machine: $MACHINE"

# Get PAT based on machine
if [ "$MACHINE" = "cbp" ]; then
    echo "📍 CBP Windows Desktop detected"
    GITHUB_PAT=$(grep GITHUB_PAT /mnt/c/exe/projects/ai-agents/.env | cut -d'=' -f2)
elif [ "$MACHINE" = "sprout" ]; then
    echo "📍 Jetson (sprout) detected"
    GITHUB_PAT=$(cat "/home/sprout/ai-workspace/github pat.txt")
else
    echo "❌ Unknown machine - check private-context/machines/"
    exit 1
fi

# Verify PAT was found
if [ -z "$GITHUB_PAT" ]; then
    echo "❌ GitHub PAT not found for machine: $MACHINE"
    exit 1
fi

echo "✅ GitHub PAT found"

# Push to GitHub
echo "🚀 Pushing to GitHub..."
git push https://dp-web4:${GITHUB_PAT}@github.com/dp-web4/ai-dna-discovery.git main

if [ $? -eq 0 ]; then
    echo "✅ Push successful!"
else
    echo "❌ Push failed"
    exit 1
fi