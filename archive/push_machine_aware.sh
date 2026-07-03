#!/bin/bash
# Machine-aware git push script (SSH — the GitHub PAT is retired)
# SSH keys are per-machine and handled by ssh-agent, so no per-machine
# token selection is needed anymore; origin is git@github.com:dp-web4/ai-dna-discovery.git

echo "🔍 Identifying machine..."
MACHINE=$(hostname)
echo "Machine: $MACHINE"

# Push to GitHub over SSH
echo "🚀 Pushing to GitHub over SSH..."
if git push origin main; then
    echo "✅ Push successful!"
else
    echo "❌ Push failed — check this machine's SSH key is loaded and added to the dp-web4 account (ssh-add -l; ssh -T git@github.com)"
    exit 1
fi
