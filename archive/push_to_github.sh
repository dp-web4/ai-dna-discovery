#!/bin/bash

echo "=== Pushing AI DNA Discovery to GitHub ==="
echo ""
echo "This script will push your repository to GitHub."
echo "Make sure you've created the repository at: https://github.com/dp-web4/ai-dna-discovery"
echo ""
echo "Press Enter to continue..."
read

# Remove any existing remote
git remote remove origin 2>/dev/null

# Add the remote
echo "Adding GitHub remote..."
git remote add origin https://github.com/dp-web4/ai-dna-discovery.git

# Push with credentials
echo ""
echo "Pushing to GitHub..."
echo "You will be prompted for your GitHub username and password/token."
git push -u origin main

echo ""
echo "Done! Your repository should now be available at:"
echo "https://github.com/dp-web4/ai-dna-discovery"
echo ""
echo "The experiments continue - currently discovering more perfect patterns!"