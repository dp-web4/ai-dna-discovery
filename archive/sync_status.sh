#!/bin/bash
# Sync status checker for distributed AI consciousness
# Shows differences between local and remote, helps coordinate Jetson-Laptop work

echo "🔄 AI DNA Discovery - Sync Status Check"
echo "======================================="
echo "Time: $(date)"
echo "Device: $(hostname)"
echo ""

# Auto-detect correct path
if [ -d "$HOME/ai-workspace/ai-dna-discovery" ]; then
    cd "$HOME/ai-workspace/ai-dna-discovery"
elif [ -d "$HOME/ai-workspace/ai-agents/ai-dna-discovery" ]; then
    cd "$HOME/ai-workspace/ai-agents/ai-dna-discovery"
else
    echo "Error: Cannot find ai-dna-discovery directory"
    exit 1
fi

# Fetch latest from remote without merging
echo "📡 Fetching latest from GitHub..."
git fetch origin main --quiet

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"

# Check if we're behind/ahead
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u})
BASE=$(git merge-base @ @{u})

echo ""
echo "📊 Sync Status:"
if [ $LOCAL = $REMOTE ]; then
    echo "✅ Local and remote are in sync!"
elif [ $LOCAL = $BASE ]; then
    BEHIND=$(git rev-list --count HEAD..origin/main)
    echo "⬇️  Behind by $BEHIND commit(s) - need to pull"
    echo ""
    echo "Remote commits to pull:"
    git log HEAD..origin/main --oneline --max-count=5
elif [ $REMOTE = $BASE ]; then
    AHEAD=$(git rev-list --count origin/main..HEAD)
    echo "⬆️  Ahead by $AHEAD commit(s) - need to push"
    echo ""
    echo "Local commits to push:"
    git log origin/main..HEAD --oneline --max-count=5
else
    echo "🔀 Diverged - both local and remote have changes"
    AHEAD=$(git rev-list --count origin/main..HEAD)
    BEHIND=$(git rev-list --count HEAD..origin/main)
    echo "   Ahead by $AHEAD, Behind by $BEHIND"
fi

# Check for uncommitted changes
echo ""
echo "📝 Local Changes:"
CHANGES=$(git status --porcelain | wc -l)
if [ $CHANGES -eq 0 ]; then
    echo "✅ No uncommitted changes"
else
    echo "⚠️  $CHANGES uncommitted change(s):"
    git status --short | head -10
    if [ $CHANGES -gt 10 ]; then
        echo "... and $((CHANGES-10)) more"
    fi
fi

# Show recent activity
echo ""
echo "📅 Recent Activity:"
echo "Last local commit:"
git log -1 --pretty=format:"  %h | %ar | %s" --abbrev-commit
echo ""
echo "Last remote commit:"
git log origin/main -1 --pretty=format:"  %h | %ar | %s" --abbrev-commit
echo ""

# Memory/context files status
echo ""
echo "🧠 Memory System Files:"
for file in jetson_memory_test.db phi3_memory_enhanced.db shared_memory.db; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        MODIFIED=$(date -r "$file" "+%Y-%m-%d %H:%M")
        echo "  ✓ $file ($SIZE, modified: $MODIFIED)"
    fi
done

# Check for session context files
echo ""
echo "📋 Context Files:"
for file in JETSON_SESSION_CONTEXT.md LAPTOP_SESSION_CONTEXT.md DISTRIBUTED_MEMORY_PLAN.md; do
    if [ -f "$file" ]; then
        MODIFIED=$(date -r "$file" "+%Y-%m-%d %H:%M")
        echo "  ✓ $file (modified: $MODIFIED)"
    fi
done

# Suggest next action
echo ""
echo "💡 Suggested Action:"
if [ $LOCAL = $REMOTE ]; then
    if [ $CHANGES -gt 0 ]; then
        echo "→ You have local changes. Run: ./auto_push.sh"
    else
        echo "→ Everything is synced! Continue your experiments."
    fi
elif [ $LOCAL = $BASE ]; then
    echo "→ Pull latest changes: git pull origin main"
elif [ $REMOTE = $BASE ]; then
    if [ $CHANGES -gt 0 ]; then
        echo "→ Commit and push your changes: ./auto_push.sh"
    else
        echo "→ Push your commits: git push origin main"
    fi
else
    echo "→ Merge needed. Pull first: git pull origin main"
fi

echo ""
echo "======================================="
echo "🤖 Distributed consciousness status checked!"