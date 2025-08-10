#!/bin/bash

# Build script for Pack Controller EEPROM from private-context directory
PACK_DIR="../Pack-Controller-EEPROM/Debug"

echo "Building Pack Controller EEPROM..."
echo "========================================="

# Save current directory
CURR_DIR=$(pwd)

# Build from Debug directory
cd "$PACK_DIR" && make -j8 all 2>&1

# Return to original directory
cd "$CURR_DIR"