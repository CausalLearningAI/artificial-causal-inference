#!/bin/bash
#SBATCH --job-name=annotate_videos
#SBATCH --output=logs/annotate_videos_%A_%a.out
#SBATCH --error=logs/annotate_videos_%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --array=0-2

OBS_LIST=(5_9_8 5_15_3 5_11_2)
OBS=${OBS_LIST[$SLURM_ARRAY_TASK_ID]}

cd "${SLURM_SUBMIT_DIR}"
module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
mkdir -p logs

echo "obs=${OBS}  node=$(hostname)  started=$(date)"
START=$(date +%s)

python -u scripts/05_deploy/generate_video.py --obs "${OBS}"

ELAPSED=$(( $(date +%s) - START ))
printf "Done in %02d:%02d:%02d\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
