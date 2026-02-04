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
    
    echo "=================================================="
    echo "Processing ${SUBJECT}/${VERSION}"
    echo "=================================================="
    
    # Step 1: Extract frames
    echo "Step 1/3: Extracting frames..."
    STEP_START=$(date +%s)
    if ! python -u src/dataset/get_frames.py experiment="$SUBJECT/$VERSION"; then
        echo "❌ Frames extraction failed for ${SUBJECT}/${VERSION}" >&2
        return 1
    fi
    STEP_END=$(date +%s)
    STEP_ELAPSED=$((STEP_END - STEP_START))
    echo "  Time: ${STEP_ELAPSED}s"
    
    # Step 2: Generate annotations
    echo ""
    echo "Step 2/3: Generating annotations..."
    STEP_START=$(date +%s)
    if ! python -u -m src.dataset.get_annotations experiment="$SUBJECT/$VERSION"; then
        echo "❌ Annotation generation failed for ${SUBJECT}/${VERSION}" >&2
        return 1
    fi
    STEP_END=$(date +%s)
    STEP_ELAPSED=$((STEP_END - STEP_START))
    echo "  Time: ${STEP_ELAPSED}s"
    
    # Step 3: Generate HF dataset
    echo ""
    echo "Step 3/3: Generating Hugging Face dataset..."
    STEP_START=$(date +%s)
    if ! python -u src/dataset/get_dataset.py experiment="$SUBJECT/$VERSION"; then
        echo "❌ HF dataset generation failed for ${SUBJECT}/${VERSION}" >&2
        return 1
    fi
    STEP_END=$(date +%s)
    STEP_ELAPSED=$((STEP_END - STEP_START))
    echo "  Time: ${STEP_ELAPSED}s"
    
    local END_TIME=$(date +%s)
    local ELAPSED=$((END_TIME - START_TIME))
    echo ""
    echo "Done! Total time for ${SUBJECT}/${VERSION}: ${ELAPSED} seconds ($(printf '%02d:%02d:%02d\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))))"
    echo ""
}

# Process experiments
process_experiment "ants" "v1"
process_experiment "ants" "v2"

process_experiment "mice" "v1"
process_experiment "mice" "v2"
