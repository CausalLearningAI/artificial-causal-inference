#!/bin/bash
#
# Standardize data for a given experiment
#
#SBATCH --job-name=standardize_data
#SBATCH --output=logs/standardize_data_%j.out
#SBATCH --error=logs/standardize_data_%j.err
#SBATCH --time=02:00:00
##SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load environment
module load conda
conda activate crl

# Disable Python output buffering for real-time logs
export PYTHONUNBUFFERED=1

# Set subject and version to process
SUBJECT="ants"

VERSION="v1"
echo "Processing: ${SUBJECT}/${VERSION}"
python -u src/data/standardize.py experiment.subject="$SUBJECT" experiment.version="$VERSION"
python -u src/data/get_metadata.py experiment.subject="$SUBJECT" experiment.version="$VERSION"
echo "Done!"

VERSION="v2"
echo "Processing: ${SUBJECT}/${VERSION}"
python -u src/data/standardize.py experiment.subject="$SUBJECT" experiment.version="$VERSION"
python -u src/data/get_metadata.py experiment.subject="$SUBJECT" experiment.version="$VERSION"
echo "Done!"

