#!/bin/bash

# Ensure the user passed the required arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run_worker.sh <GPU_ID> <RANK> <WORLD_SIZE>"
    exit 1
fi

GPU=$1
RANK=$2
WORLD_SIZE=$3

# Infinite loop that only breaks on a clean exit (Exit Code 0)
while true; do
    echo "============================================================"
    echo "Launching worker on GPU $GPU (Rank $RANK)..."
    echo "============================================================"
    
    python audio_next.py --gpu $GPU --rank $RANK --world_size $WORLD_SIZE --batch_size 1
    
    EXIT_CODE=$?
    
    # Exit Code 0 means the Python script hit the end of the dataset naturally
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Worker on GPU $GPU finished all files successfully! Exiting."
        break
    else
        echo "------------------------------------------------------------"
        echo "Worker on GPU $GPU crashed (Exit Code $EXIT_CODE). CUDA collapse detected."
        echo "Bad file recorded. Restarting environment in 3 seconds..."
        echo "------------------------------------------------------------"
        sleep 3
    fi
done