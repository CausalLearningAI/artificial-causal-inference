#!/bin/bash
#
# Standardize data for a given experiment
#
#SBATCH --job-name=get_data
#SBATCH --output=logs/get_data_%j.out
#SBATCH --error=logs/get_data_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
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
    
    echo "Processing: ${SUBJECT}/${VERSION}"
    python -u src/data/standardize.py experiment="$SUBJECT/$VERSION"
    python -u src/data/get_metadata.py experiment="$SUBJECT/$VERSION"
    
    local END_TIME=$(date +%s)
    local ELAPSED=$((END_TIME - START_TIME))
    echo "Done! Time elapsed for ${SUBJECT}/${VERSION}: ${ELAPSED} seconds ($(printf '%02d:%02d:%02d\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))))"
}

# Runs for different experiments
# process_experiment "ants" "v1"
# process_experiment "ants" "v2"
process_experiment "ants" "v3"
process_experiment "ants" "v4"

# process_experiment "mice" "v1"
# process_experiment "mice" "v2"

