#!/bin/bash
#
# POV embeddings of one identity with the inputs staged on node-local disk: annotations.csv and
# frames/pov/<identity> are streamed from NFS once (one tar per video, STAGE_STREAMS in parallel), get_embeddings.py runs against the
# local copy (paths.datasets_dir), and the embedding directory is copied back at the end.
# Logs GPU utilisation every 60 s to logs/gpu_<job>_<id>.csv.
#
# Usage: VERSION=v4_infested IDENT=blue sbatch scripts/02_embeddings/job_pov_staged.sh
#
#SBATCH --job-name=emb_pov
#SBATCH --output=logs/emb_pov_%j.out
#SBATCH --error=logs/emb_pov_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
ENVS=${CONDA_ENVS:-$HOME/.conda/envs}
export PATH="$ENVS/crl/bin:$PATH" PYTHONUNBUFFERED=1
V=${VERSION:?}; ID=${IDENT:?}; ENC=${ENCODER:-dinov2}; TOK=${TOKEN:-class}
S=/localhome/$USER/$SLURM_JOB_ID; mkdir -p "$S" 2>/dev/null || S=${TMPDIR:-/tmp}/$USER/$SLURM_JOB_ID; mkdir -p "$S"
trap 'kill ${GPUMON:-} 2>/dev/null || true; rm -rf "$S"' EXIT
export HF_DATASETS_CACHE=$S/hf_cache
echo "node=$(hostname) stage=$S $(nvidia-smi --query-gpu=name --format=csv,noheader)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 60 > logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv & GPUMON=$!

t=$(date +%s)
L=$S/dataset/ants/$V; mkdir -p "$L"
# NFS is latency-bound (~30 files/s per stream when busy): one tar stream per video, many in parallel
cp dataset/ants/$V/annotations.csv "$L/"; mkdir -p "$L/frames/pov/$ID"
ls -f dataset/ants/$V/frames/pov/$ID | grep -v '^\.' | head -n ${MAX_VIDEOS:-100000} |
    xargs -P ${STAGE_STREAMS:-24} -I@ sh -c "tar -C dataset/ants/$V -cf - frames/pov/$ID/@ | tar -C '$L' -xf -"
echo "[stage-in $(( $(date +%s) - t ))s] $(ls $L/frames/pov/$ID | wc -l) observations, $(df -h --output=used $S | tail -1) used"

t=$(date +%s)
python -u src/embedding/get_embeddings.py experiment=ants/$V encoder=$ENC token=$TOK \
    batch_size=192 num_workers=$((SLURM_CPUS_PER_TASK - 2)) device=cuda +frame_type=pov +pov_identity=$ID \
    overwrite.embeddings=false +paths.datasets_dir=$S/dataset 2> >(grep -v "frame/s\]\|it/s\]" >&2)
echo "[embed $(( $(date +%s) - t ))s]"

t=$(date +%s)
OUT=dataset/ants/$V/embeddings/pov/$ID/$ENC/$TOK
mkdir -p "$OUT"
cp -r "$L/embeddings/pov/$ID/$ENC/$TOK/." "$OUT/"
rm -f "$OUT/embeddings.npy.tmp"
echo "[copy-back $(( $(date +%s) - t ))s] -> $OUT"; ls -la "$OUT"
