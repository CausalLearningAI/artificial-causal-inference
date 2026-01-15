#!/bin/bash
#
# Extract frames from observation videos
#
#SBATCH --job-name=get_frames
#SBATCH --output=logs/get_frames_%j.out
#SBATCH --error=logs/get_frames_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=visualization
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load environment
module load conda
conda activate crl

# Disable Python output buffering for real-time logs
export PYTHONUNBUFFERED=1

# Function to process experiment and track time
process_experiment() {
    local SUBJECT=$1
    local VERSION=$2
    local START_TIME=$(date +%s)
    
    echo "Extracting frames: ${SUBJECT}/${VERSION}"
    python -u src/dataset/get_frames.py experiment.subject="$SUBJECT" experiment.version="$VERSION"
    
    local END_TIME=$(date +%s)
    local ELAPSED=$((END_TIME - START_TIME))
    echo "Done! Time elapsed for ${SUBJECT}/${VERSION}: ${ELAPSED} seconds ($(printf '%02d:%02d:%02d\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))))"
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
