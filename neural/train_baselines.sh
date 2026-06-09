#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the valid levels
LEVELS=("L1" "L2" "L3" "L4" "L5")

# Loop through each level and execute the training script
for LVL in "${LEVELS[@]}"; do
    echo "========================================="
    echo " Starting training for level: ${LVL} "
    echo "========================================="
    
    # Call your python script (assuming it uses the --level flag we set up earlier)
    python train_modern_dit.py --level "$LVL" --device "cuda:2"
    
    echo " Finished training for level: ${LVL}"
    echo "========================================="
    echo ""
done

echo "All training runs completed successfully!"