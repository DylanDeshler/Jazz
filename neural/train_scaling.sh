#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Help message function
print_usage() {
    echo "Usage: $0 <device> <mode> [start_level]"
    echo "  <device>      : e.g., cuda:0, cpu"
    echo "  <mode>        : baseline | measure"
    echo "  [start_level] : (Optional) L1, L2, L3, L4, or L5. Defaults to L1."
    echo "Example       : $0 cuda:0 measure L3"
}

# Check if required arguments were provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing required arguments."
    print_usage
    exit 1
fi

DEVICE=$1
MODE=$2
START_LEVEL=${3:-"L1"} # Defaults to L1 if not provided

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

# Full list of levels
LEVELS=("L1" "L2" "L3" "L4" "L5" "L6")

# Validate the start level and find its index
START_INDEX=-1
for i in "${!LEVELS[@]}"; do
    if [ "${LEVELS[$i]}" = "$START_LEVEL" ]; then
        START_INDEX=$i
        break
    fi
done

if [ "$START_INDEX" -eq -1 ]; then
    echo "Error: Invalid start level '$START_LEVEL'. Must be one of: L1, L2, L3, L4, L5"
    print_usage
    exit 1
fi

echo "Target Script : ${SCRIPT_TO_RUN}"
echo "Target Device : ${DEVICE}"
echo "Target Mode   : ${MODE}"
echo "Starting At   : ${START_LEVEL}"
echo ""

# Loop through each level starting from the requested index
for ((i=START_INDEX; i<${#LEVELS[@]}; i++)); do
    LVL=${LEVELS[$i]}
    
    echo "========================================="
    echo " Starting training for level: ${LVL} "
    echo "========================================="
    
    # Executes the selected script with the parameters
    python "$SCRIPT_TO_RUN" --level "$LVL" --device "$DEVICE"
    
    echo " Finished training for level: ${LVL}"
    echo "========================================="
    echo ""
done

echo "All requested training runs completed successfully using ${SCRIPT_TO_RUN} on ${DEVICE}!"