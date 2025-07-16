#!/bin/bash
# Script to rename files with colons to use underscores instead
# This fixes Windows compatibility issues

echo "Fixing filenames with colons for Windows compatibility..."

# Find all files with colons in embedding_space_results directory
for file in embedding_space_results/*:*.png; do
    if [ -f "$file" ]; then
        # Replace : with _
        newname=$(echo "$file" | sed 's/:/_/g')
        echo "Renaming: $file -> $newname"
        mv "$file" "$newname"
    fi
done

echo "Done! All colons replaced with underscores."