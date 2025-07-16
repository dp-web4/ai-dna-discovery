#!/bin/bash
# Run the language evolution experiment with output logging

echo "Starting Common Language Evolution Experiment"
echo "============================================"
echo "Start time: $(date)"
echo "This experiment will run for approximately 50-60 minutes"
echo "Progress will be saved every 10 rounds"
echo ""

# Create log file
LOG_FILE="common_language_results/evolution_log_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p common_language_results

# Run experiment with output to both console and log file
python3 common_language_evolution.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "Experiment completed at: $(date)"
echo "Log saved to: $LOG_FILE"