#!/bin/bash
#
# Extract frames and generate annotations for datasets
#
#SBATCH --job-name=get_dataset
#SBATCH --output=logs/get_dataset_%j.out
#SBATCH --error=logs/get_dataset_%j.err
#SBATCH --time=04:00:00
##SBATCH --partition=visualization
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load environment
module load conda
conda activate crl

# Disable Python output buffering for real-time logs
export PYTHONUNBUFFERED=1

# Fail fast on any error
set -euo pipefail

# Function to process experiment and track time
process_experiment() {
    local SUBJECT=$1
    local VERSION=$2
    local START_TIME=$(date +%s)
    
    echo "=========================================="
    echo "EXPERIMENT: ${SUBJECT}/${VERSION}"
    echo "=========================================="
    
    # Extract frames
    STEP_START=$(date +%s)
    if python -u src/dataset/get_frames.py experiment="$SUBJECT/$VERSION"; then
        STEP_ELAPSED=$(($(date +%s) - STEP_START))
        echo "[Extract frames completed in ${STEP_ELAPSED}s]"
    else
        echo "[Extract frames FAILED]"
        return 1
    fi
    echo ""
    
    # Generate annotations
    STEP_START=$(date +%s)
    if python -u -m src.dataset.get_annotations experiment="$SUBJECT/$VERSION"; then
        STEP_ELAPSED=$(($(date +%s) - STEP_START))
        echo "[Generate annotations completed in ${STEP_ELAPSED}s]"
    else
        echo "[Generate annotations FAILED]"
        return 1
    fi
    echo ""
    
    # Generate HF dataset
    STEP_START=$(date +%s)
    if python -u src/dataset/get_dataset.py experiment="$SUBJECT/$VERSION"; then
        STEP_ELAPSED=$(($(date +%s) - STEP_START))
        echo "[Generate HF dataset completed in ${STEP_ELAPSED}s]"
    else
        echo "[Generate HF dataset FAILED]"
        return 1
    fi
    echo ""
    
    local ELAPSED=$(($(date +%s) - START_TIME))
    echo "=========================================="
    printf "TOTAL TIME: %02d:%02d:%02d\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
    echo "=========================================="
    echo ""
}

# Process experiments
# process_experiment "ants" "v1"
# process_experiment "ants" "v2"
process_experiment "ants" "v3"
# process_experiment "ants" "v4"

# process_experiment "mice" "v1"
# process_experiment "mice" "v2"
