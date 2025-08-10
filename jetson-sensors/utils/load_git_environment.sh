#!/bin/bash
#
# Load Git Environment on Startup
# This script ensures the PAT is available in the environment
#

# Define paths
PAT_SOURCE="/home/sprout/ai-workspace/github pat.txt"
ENV_FILE="/home/sprout/ai-workspace/.env"

# Check if PAT source exists
if [ ! -f "$PAT_SOURCE" ]; then
    echo "[$(date)] ERROR: PAT source not found at '$PAT_SOURCE'" >&2
    exit 1
fi

# Create/update .env file
PAT=$(cat "$PAT_SOURCE")
echo "GITHUB_PAT=${PAT}" > "$ENV_FILE"

# Export for current session
export GITHUB_PAT="${PAT}"

# Log success
echo "[$(date)] Git environment loaded successfully" 

# Optionally set git credential helper to use the PAT
git config --global credential.helper store
git config --global user.name "dp-web4"
git config --global user.email "dp@web4.ai"

echo "[$(date)] Git configuration updated"