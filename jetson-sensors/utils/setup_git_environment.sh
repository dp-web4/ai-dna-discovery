#!/bin/bash
#
# Setup Git Environment Script
# Ensures PAT is available and repositories can be cloned/pushed
#

echo "Setting up Git environment..."

# Check if PAT source file exists
if [ ! -f "../github pat.txt" ]; then
    echo "ERROR: PAT source file not found at '../github pat.txt'"
    echo "Please ensure the file exists with your GitHub Personal Access Token"
    exit 1
fi

# Create or update .env file
echo "Creating/updating .env file..."
PAT=$(cat "../github pat.txt")
echo "GITHUB_PAT=${PAT}" > ../.env
echo "✓ .env file created at ../.env"

# Verify PAT format
if [[ $PAT == github_pat_* ]]; then
    echo "✓ PAT format looks correct"
else
    echo "WARNING: PAT doesn't start with 'github_pat_' - verify it's correct"
fi

# Source the .env for current session
source ../.env

# Test authentication
echo ""
echo "Testing GitHub authentication..."
if git ls-remote https://dp-web4:${GITHUB_PAT}@github.com/dp-web4/private-context.git HEAD &>/dev/null; then
    echo "✓ GitHub authentication successful!"
else
    echo "✗ GitHub authentication failed - check your PAT"
    exit 1
fi

echo ""
echo "Git environment setup complete!"
echo ""
echo "You can now:"
echo "  • Clone private repos: git clone https://dp-web4:\${GITHUB_PAT}@github.com/dp-web4/REPO.git"
echo "  • Push to repos: git push https://dp-web4:\${GITHUB_PAT}@github.com/dp-web4/REPO.git main"
echo ""
echo "Available repositories:"
echo "  • private-context (documentation & consciousness bridge)"
echo "  • web4 (Web4 implementation project)"
echo ""
echo "Clone repositories to /home/sprout/ai-workspace/:"
echo "  cd /home/sprout/ai-workspace"
echo "  git clone https://dp-web4:\${GITHUB_PAT}@github.com/dp-web4/web4.git"
echo ""
echo "PAT is available in environment as: \$GITHUB_PAT"