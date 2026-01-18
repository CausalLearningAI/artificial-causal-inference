#!/bin/bash
#
# Extract frames and generate annotations for datasets
#
#SBATCH --job-name=datasets
#SBATCH --output=logs/get_datasets_%j.out
#SBATCH --error=logs/get_datasets_%j.err
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
    echo "Step 1/2: Extracting frames..."
    if ! python -u src/dataset/get_frames.py experiment.subject="$SUBJECT" experiment.version="$VERSION"; then
        echo "❌ Frames extraction failed for ${SUBJECT}/${VERSION}" >&2
        return 1
    fi
    
    # Step 2: Generate annotations
    echo ""
    echo "Step 2/2: Generating annotations..."
    if ! python -u -m src.dataset.get_annotations --subject "$SUBJECT" --version "$VERSION"; then
        echo "❌ Annotation generation failed for ${SUBJECT}/${VERSION}" >&2
        return 1
    fi
    
    local END_TIME=$(date +%s)
    local ELAPSED=$((END_TIME - START_TIME))
    echo ""
    echo "Done! Total time for ${SUBJECT}/${VERSION}: ${ELAPSED} seconds ($(printf '%02d:%02d:%02d\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))))"
    echo ""
}

# Process ants
SUBJECT="ants"

VERSION="v1"
process_experiment "$SUBJECT" "$VERSION"

VERSION="v2"
process_experiment "$SUBJECT" "$VERSION"

# Process mice
# SUBJECT="mice"

# VERSION="v1"
# process_experiment "$SUBJECT" "$VERSION"
