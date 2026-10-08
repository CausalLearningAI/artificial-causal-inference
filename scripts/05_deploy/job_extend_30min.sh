#!/bin/bash
#
# One observation of the 30-min extension per array task (extend_30min.py extend-obs):
# standardize, JPGs >= N_OLD, tracking of the new video, POV crops >= N_OLD, staged in /localhome.
# ffmpeg comes from FFMPEG_ENV (6.1.1 + openh264 2.1.1, the build that made the 10-min clips,
# so frames 0..N_OLD-1 come out bit-identical); python from the crl env.
#
# Usage:
#   VERSION=v5 sbatch --array=0-189%30 scripts/05_deploy/job_extend_30min.sh
#
#SBATCH --job-name=ext30
#SBATCH --output=logs/ext30_%A_%a.out
#SBATCH --error=logs/ext30_%A_%a.err
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
"$ENVS/crl/bin/python" -u scripts/05_deploy/extend_30min.py --version "${VERSION:-v5}" extend-obs
