#!/bin/bash
# Repository cleanup script for AI DNA Discovery
# This will remove virtual environments and cache files

echo "=== AI DNA Discovery Repository Cleanup ==="
echo "Current repository size: $(du -sh . | cut -f1)"
echo "Current file count: $(find . -type f | wc -l)"
echo ""

# Safety check
read -p "This will remove all virtual environments and cache files. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

echo "Removing virtual environments..."
# Remove specific known virtual environments
rm -rf ./dictionary/phoenician_env
rm -rf ./model-training/unsloth_env  
rm -rf ./sensor-consciousness/sensor_venv

# Remove any other potential virtual environments
find . -type d -name "*_env" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null
find . -type d -name "*_venv" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null
find . -type d -name "venv" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null
find . -type d -name "env" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null

echo "Removing Python cache files..."
find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
find . -name "*.pyd" -delete 2>/dev/null

echo "Removing other temporary files..."
find . -name ".DS_Store" -delete 2>/dev/null
find . -name "Thumbs.db" -delete 2>/dev/null
find . -name "*.swp" -delete 2>/dev/null
find . -name "*.swo" -delete 2>/dev/null

echo ""
echo "=== Cleanup Complete ==="
echo "New repository size: $(du -sh . | cut -f1)"
echo "New file count: $(find . -type f | wc -l)"
echo ""
echo "Remember to:"
echo "1. Review and update .gitignore"
echo "2. Run 'git add -A' to stage deletions"
echo "3. Commit with a descriptive message"
echo "4. Consider using Git LFS for large model files"