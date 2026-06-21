#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Help message function
print_usage() {
    echo "Usage: $0 <device> <mode> <axis> [start_level] [chunks]"
    echo "  <device>      : e.g., cuda:0, cpu"
    echo "  <mode>        : baseline | measure"
    echo "  <axis>        : depth | width"
    echo "  [start_level] : (Optional) L0, L1, L2, L3, L4, or L5. Defaults to L0."
    echo "  [chunks]      : (Optional) Number of chunks. Defaults to 32."
    echo "Example       : $0 cuda:0 measure depth L3 64"
}

# Check if required arguments were provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: Missing required arguments."
    print_usage
    exit 1
fi

DEVICE=$1
MODE=$2
AXIS=$3
START_LEVEL=${4:-"L0"} # Defaults to L0 if not provided
CHUNKS=${5:-32}        # Defaults to 32 if not provided

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

# Validate the axis argument
if [ "$AXIS" != "depth" ] && [ "$AXIS" != "width" ]; then
    echo "Error: Invalid axis '$AXIS'. Must be 'depth' or 'width'."
    print_usage
    exit 1
fi

# Full list of levels
LEVELS=("L0" "L1" "L2" "L3" "L4" "L5")

# Validate the start level and find its index
START_INDEX=-1
for i in "${!LEVELS[@]}"; do
    if [ "${LEVELS[$i]}" = "$START_LEVEL" ]; then
        START_INDEX=$i
        break
    fi
done

if [ "$START_INDEX" -eq -1 ]; then
    echo "Error: Invalid start level '$START_LEVEL'. Must be one of: L0, L1, L2, L3, L4, L5"
    print_usage
    exit 1
fi

echo "Target Script : ${SCRIPT_TO_RUN}"
echo "Target Device : ${DEVICE}"
echo "Target Mode   : ${MODE}"
echo "Target Axis   : ${AXIS}"
echo "Starting At   : ${START_LEVEL}"
echo "Chunks Count  : ${CHUNKS}"
echo ""

# Loop through each level starting from the requested index
for ((i=START_INDEX; i<${#LEVELS[@]}; i++)); do
    LVL=${LEVELS[$i]}
    
    echo "========================================="
    echo " Starting training for level: ${LVL} "
    echo "========================================="
    
    # Executes the selected script with the parameters, including --axis and --n_chunks
    python "$SCRIPT_TO_RUN" --level "$LVL" --device "$DEVICE" --axis "$AXIS" --n_chunks "$CHUNKS"
    
    echo " Finished training for level: ${LVL}"
    echo "========================================="
    echo ""
done

echo "All requested training runs completed successfully using ${SCRIPT_TO_RUN} on ${DEVICE}!"