#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Help message function
print_usage() {
    echo "Usage: $0 <device> <mode> <axis> [start_level] [n_chunks]"
    echo "  <device>      : e.g., cuda:0, cpu"
    echo "  <mode>        : baseline | measure"
    echo "  <axis>        : depth | width | chunks"
    echo "  [start_level] : (Optional) L0, L1, L2, L3, L4, or L5. Defaults to L0 (or L3 if axis is chunks)."
    echo "  [n_chunks]    : (Optional) Static chunk size if axis is depth/width. Defaults to 32."
    echo "Example       : $0 cuda:0 measure chunks"
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

# Determine defaults based on the selected axis
if [ "$AXIS" = "chunks" ]; then
    START_LEVEL=${4:-"L3"}    # Defaults to L3 for chunks axis
    N_CHUNKS=${5:-32}         # Default fall-back if needed, overridden by loop sequence
else
    START_LEVEL=${4:-"L0"}    # Defaults to L0 for depth/width
    N_CHUNKS=${5:-32}         # Hard defaults to 32 if not provided
fi

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
if [ "$AXIS" != "depth" ] && [ "$AXIS" != "width" ] && [ "$AXIS" != "chunks" ]; then
    echo "Error: Invalid axis '$AXIS'. Must be 'depth', 'width', or 'chunks'."
    print_usage
    exit 1
fi

echo "Target Script : ${SCRIPT_TO_RUN}"
echo "Target Device : ${DEVICE}"
echo "Target Mode   : ${MODE}"
echo "Target Axis   : ${AXIS}"

# --- Execution Logic Breakdown ---

if [ "$AXIS" = "chunks" ]; then
    # Define the chunk loop sequence
    # CHUNK_ITEMS=(8 16 24 32 40 48)
    CHUNK_ITEMS=(8 16 24 40 48)
    
    echo "Fixed Level   : ${START_LEVEL}"
    echo "Cycling Chunks: ${CHUNK_ITEMS[*]}"
    echo ""

    for CURRENT_CHUNKS in "${CHUNK_ITEMS[@]}"; do
        echo "========================================="
        echo " Starting training for n_chunks: ${CURRENT_CHUNKS} (Level: ${START_LEVEL}) "
        echo "========================================="
        
        # Executes python script sweeping across n_chunks array
        python "$SCRIPT_TO_RUN" --level "$START_LEVEL" --device "$DEVICE" --axis "$AXIS" --n_chunks "$CURRENT_CHUNKS"
        
        echo " Finished training for n_chunks: ${CURRENT_CHUNKS}"
        echo "========================================="
        echo ""
    done
else
    # Traditional level-based loop logic for depth / width
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

    echo "Starting At   : ${START_LEVEL}"
    echo "Chunks Count  : ${N_CHUNKS}"
    echo ""

    for ((i=START_INDEX; i<${#LEVELS[@]}; i++)); do
        LVL=${LEVELS[$i]}
        
        echo "========================================="
        echo " Starting training for level: ${LVL} "
        echo "========================================="
        
        # Passes static n_chunks value down to python
        python "$SCRIPT_TO_RUN" --level "$LVL" --device "$DEVICE" --axis "$AXIS" --n_chunks "$N_CHUNKS"
        
        echo " Finished training for level: ${LVL}"
        echo "========================================="
        echo ""
    done
fi

echo "All requested training runs completed successfully using ${SCRIPT_TO_RUN} on ${DEVICE}!"