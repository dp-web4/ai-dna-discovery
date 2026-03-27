#!/bin/bash

# Start Continuous AI DNA Discovery Experiments
# This script ensures experiments run continuously

echo "Starting Continuous Experiment Engine..."
echo "Searching for Universal AI Embedding Language (AI DNA)"
echo "This will run indefinitely. Use Ctrl+C to stop."

# Create results directory
mkdir -p /home/dp/ai-workspace/ai_dna_results

# Run the engine
while true; do
    echo "Starting experiment engine at $(date)"
    python3 /home/dp/ai-workspace/continuous_experiment_engine.py
    
    # If it exits, wait a bit and restart
    echo "Engine stopped. Restarting in 10 seconds..."
    sleep 10
done