#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if a device argument was provided
if [ -z "$1" ]; then
    echo "Error: No device specified."
    echo "Usage: $0 <device>  (e.g., $0 cuda:0)"
    exit 1
fi

DEVICE=$1
LEVELS=("L1" "L2" "L3" "L4" "L5")

echo "Starting training sequence on device: ${DEVICE}"
echo ""

# Loop through each level and execute the training script
for LVL in "${LEVELS[@]}"; do
    echo "========================================="
    echo " Starting training for level: ${LVL} "
    echo "========================================="
    
    # Passes both the current level and the chosen device to your script
    python train_modern_dit.py --level "$LVL" --device "$DEVICE"
    
    echo " Finished training for level: ${LVL}"
    echo "========================================="
    echo ""
done

echo "All training runs completed successfully on ${DEVICE}!"