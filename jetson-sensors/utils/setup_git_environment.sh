#!/bin/bash
#
# Setup Git Environment Script
# Verifies SSH access to GitHub. The GitHub PAT is retired — the ecosystem
# uses SSH only. This script no longer provisions a token; it confirms this
# machine's SSH key can reach the dp-web4 repos.
#

echo "Setting up Git environment (SSH)..."

# Test SSH authentication against a known private repo
echo ""
echo "Testing GitHub SSH authentication..."
if git ls-remote git@github.com:dp-web4/private-context.git HEAD &>/dev/null; then
    echo "✓ GitHub SSH authentication successful!"
else
    echo "✗ GitHub SSH authentication failed"
    echo "  This machine's SSH key must be generated and added to the dp-web4 GitHub account:"
    echo "    ssh-keygen -t ed25519 -C \"$(hostname)\"   # if no key exists yet"
    echo "    cat ~/.ssh/id_ed25519.pub                 # add this at github.com/settings/keys"
    echo "    ssh -T git@github.com                      # verify"
    exit 1
fi

echo ""
echo "Git environment setup complete!"
echo ""
echo "You can now:"
echo "  • Clone private repos: git clone git@github.com:dp-web4/REPO.git"
echo "  • Push to repos:       git push origin main"
echo ""
echo "Available repositories:"
echo "  • private-context (documentation & consciousness bridge)"
echo "  • web4 (Web4 implementation project)"
echo ""
echo "Clone repositories to ~/ai-workspace/:"
echo "  cd ~/ai-workspace"
echo "  git clone git@github.com:dp-web4/web4.git"
