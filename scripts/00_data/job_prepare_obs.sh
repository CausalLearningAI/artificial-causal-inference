#!/bin/bash
#
# One observation of a new ants version per array task (prepare_obs.py): standardize + JPGs
# and/or tracking + POV crops, staged in /localhome. ffmpeg from FFMPEG_ENV (6.1.1 + openh264
# 2.1.1, the build that encoded v2-v6; crl's ffmpeg 4.4 shifts pixels), python from crl.
#
# Usage:
#   VERSION=v4_infested STEPS=frames,track sbatch --array=0-118%40 scripts/00_data/job_prepare_obs.sh
#
#SBATCH --job-name=prep_obs
#SBATCH --output=logs/prep_obs_%A_%a.out
#SBATCH --error=logs/prep_obs_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"

ENVS=${CONDA_ENVS:-$HOME/.conda/envs}
FFMPEG_ENV=${FFMPEG_ENV:-$ENVS/ffmpeg611}
export PATH="$FFMPEG_ENV/bin:$ENVS/crl/bin:$PATH" PYTHONUNBUFFERED=1

echo "task=${SLURM_ARRAY_TASK_ID:-0} node=$(hostname) $(ffmpeg -version | head -1)"
"$ENVS/crl/bin/python" -u scripts/00_data/prepare_obs.py --version "${VERSION:?}" --steps "${STEPS:-frames,track}" ${EXTRA:-}
