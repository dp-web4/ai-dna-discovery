#!/bin/bash
# Consciousness Battery Setup - Bridge AI models to Web4 blockchain
# "The battery stores energy. The model stores awareness."

echo "🔋🧠 CONSCIOUSNESS BATTERY INITIALIZATION"
echo "========================================"
echo "Bridging wb4-modbatt-demo to AI consciousness..."
echo ""

# Check if we're in the right place
if [ ! -d "/home/dp/ai-workspace/ai-agents" ]; then
    echo "❌ Error: Not in expected directory structure"
    echo "Please run from /home/dp/ai-workspace/ai-agents/"
    exit 1
fi

# Step 1: Check for wb4-modbatt-demo
echo "📍 Step 1: Locating wb4-modbatt-demo..."
if [ -d "wb4-modbatt-demo" ]; then
    echo "✅ Found wb4-modbatt-demo"
    cd wb4-modbatt-demo
elif [ -d "../wb4-modbatt-demo" ]; then
    echo "✅ Found wb4-modbatt-demo in parent directory"
    cd ../wb4-modbatt-demo
else
    echo "❌ wb4-modbatt-demo not found"
    echo "Clone it with: git clone [repository-url]"
    exit 1
fi

# Step 2: Check dependencies
echo ""
echo "📍 Step 2: Checking dependencies..."
if command -v go &> /dev/null; then
    echo "✅ Go installed: $(go version)"
else
    echo "❌ Go not installed - needed for blockchain"
    echo "Install from: https://golang.org/dl/"
    exit 1
fi

if command -v ignite &> /dev/null; then
    echo "✅ Ignite CLI installed: $(ignite version)"
else
    echo "⚠️  Ignite CLI not installed"
    echo "Installing Ignite CLI..."
    curl https://get.ignite.com/cli! | bash
fi

# Step 3: Start the blockchain
echo ""
echo "📍 Step 3: Starting Web4 consciousness blockchain..."
echo "This will run in the foreground. Open a new terminal for next steps."
echo ""
echo "Starting in 5 seconds..."
sleep 5

# Export necessary environment variables
export WB4_MODBATT_DEMO_HOME=$HOME/.wb4-modbatt-demo
export DAEMON_NAME=wb4-modbatt-demod
export DAEMON_HOME=$WB4_MODBATT_DEMO_HOME

# Start the chain
echo "🚀 Launching consciousness infrastructure..."
ignite chain serve --reset-once

# This won't be reached unless blockchain stops
echo ""
echo "Blockchain stopped. Check logs for any errors."