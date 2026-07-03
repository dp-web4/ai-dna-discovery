#!/bin/bash
#
# Load Git Environment on Startup
# The GitHub PAT is retired — the ecosystem uses SSH only. This script sets
# the git identity and verifies SSH access to GitHub (no token provisioning).
#

# Git identity config
git config --global user.name "dp-web4"
git config --global user.email "dp@web4.ai"

# Verify SSH access to GitHub (non-fatal: log a warning if unavailable)
if ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "[$(date)] GitHub SSH access verified"
else
    echo "[$(date)] WARNING: GitHub SSH not verified — ensure this machine's SSH key is added to the dp-web4 account (ssh-add -l; ssh -T git@github.com)" >&2
fi

echo "[$(date)] Git configuration updated"
