#!/bin/bash
#
# Metadata, per-frame annotations and the HF dataset of one version, after the per-video
# frames step (scripts/00_data/job_prepare_obs.sh). ffprobe from FFMPEG_ENV, python from crl;
# the HF cache lives on node-local disk and is removed on exit.
#
# Usage: VERSION=v4_infested sbatch scripts/01_dataset/job_dataset.sh
#
#SBATCH --job-name=dataset
#SBATCH --output=logs/dataset_%j.out
#SBATCH --error=logs/dataset_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
ENVS=${CONDA_ENVS:-$HOME/.conda/envs}
export PATH="${FFMPEG_ENV:-$ENVS/ffmpeg611}/bin:$ENVS/crl/bin:$PATH" PYTHONUNBUFFERED=1
S=/localhome/$USER/${SLURM_JOB_ID:-local}; mkdir -p "$S" 2>/dev/null || S=${TMPDIR:-/tmp}/$USER/${SLURM_JOB_ID:-local}; mkdir -p "$S"; trap 'rm -rf "$S"' EXIT
export HF_DATASETS_CACHE=$S/hf_cache
PY=$ENVS/crl/bin/python
V=${VERSION:?}
echo "node=$(hostname) version=$V"
for step in "src/data/get_metadata.py" "-m src.dataset.get_annotations" "src/dataset/get_dataset.py"; do
    t=$(date +%s)
    $PY -u $step experiment=ants/$V 2> >(grep -v "%|" >&2)
    echo "[$step done in $(( $(date +%s) - t ))s]"
done
