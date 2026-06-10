#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Help message function
print_usage() {
    echo "Usage: $0 <device> <mode>"
    echo "  <device> : e.g., cuda:0, cpu"
    echo "  <mode>   : baseline | measure"
    echo "Example  : $0 cuda:0 measure"
}

# Check if both arguments were provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing required arguments."
    print_usage
    exit 1
fi

DEVICE=$1
MODE=$2

# Determine which script to run based on the mode
if [ "$MODE" = "baseline" ]; then
    SCRIPT_TO_RUN="train_modern_dit.py"
elif [ "$MODE" = "measure" ]; then
    SCRIPT_TO_RUN="train_modern_dit_measures.py"
else
    echo "Error: Invalid mode '$MODE'. Must be 'baseline' or 'measure'."
    print_usage
    exit 1
fi

LEVELS=("L1" "L2" "L3" "L4" "L5")

echo "Target Script : ${SCRIPT_TO_RUN}"
echo "Target Device : ${DEVICE}"
echo "Target Mode   : ${MODE}"
echo ""

# Loop through each level and execute the target script
for LVL in "${LEVELS[@]}"; do
    echo "========================================="
    echo " Starting training for level: ${LVL} "
    echo "========================================="
    
    # Executes the selected script with the parameters
    python "$SCRIPT_TO_RUN" --level "$LVL" --device "$DEVICE"
    
    echo " Finished training for level: ${LVL}"
    echo "========================================="
    echo ""
done

echo "All training runs completed successfully using ${SCRIPT_TO_RUN} on ${DEVICE}!"