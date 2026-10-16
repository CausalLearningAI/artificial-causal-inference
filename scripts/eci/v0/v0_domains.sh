#!/bin/bash
#
# ECI v0 (original ECI paper recipe) on one more atlas domain: frames (2 fps) -> DINOv2-S tokens -> TopK SAE ->
# mean-pooled codes -> single-pass t-tests + gate checks. See scripts/eci/v0/v0_domains.py.
#
#   DOMAIN=ants_v3 sbatch scripts/eci/v0/v0_domains.sh      (ants_v3 | ants_v4 | mice_v1 | frogs_v1)
#   DOMAIN=frogs_v1 SMOKE=2 EXTRA_ARGS="--epochs 1" OUT_TAG=_smoke sbatch --time=00:30:00 scripts/eci/v0/v0_domains.sh
#
# Big intermediates: /mnt/beegfs/locatgrp/shared/rcadei/eci_v0/<domain><OUT_TAG>/; small results:
# results/vision/<subject>/eci_v0_<domain><OUT_TAG>/; GPU log: logs/eci_v0_<domain>_<job>_gpu.csv
#
#SBATCH --job-name=eci_v0_dom
#SBATCH --output=logs/eci_v0_dom_%j.out
#SBATCH --error=logs/eci_v0_dom_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="L40S|A10|A40|A100"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail

: "${DOMAIN:?set DOMAIN}"
OUT_TAG=${OUT_TAG:-}
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd "$REPO"
SUBJ=${DOMAIN%%_*}
RES=results/vision/$SUBJ/eci_v0_${DOMAIN}${OUT_TAG}
mkdir -p logs "$RES"

STAGE=/localhome/$USER/$SLURM_JOB_ID
mkdir -p "$STAGE/videos" "$STAGE/out" "$STAGE/tok"
trap 'rm -rf "$STAGE"; kill $SMI_PID 2>/dev/null || true' EXIT
echo "node $(hostname), local disk free: $(df -h /localhome | tail -1)"

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 60 \
    > "logs/eci_v0_${DOMAIN}${OUT_TAG}_${SLURM_JOB_ID}_gpu.csv" &
SMI_PID=$!

python - "$DOMAIN" <<'PY' > "$STAGE/videos.txt"
import sys
sys.path.insert(0, 'scripts/eci/v0')
from v0_domains import load_design
d, _ = load_design(sys.argv[1])
print('\n'.join(d.video_rel))
PY
SMOKE=${SMOKE:-0}
VDIR="$STAGE/videos"
if [ "$SMOKE" -gt 0 ]; then  # test run on a few videos: read them from NFS directly
    VDIR=""; EXTRA_ARGS="--smoke $SMOKE ${EXTRA_ARGS:-}"
else
    while read -r f; do cp "$f" "$STAGE/videos/"; done < "$STAGE/videos.txt"
    echo "staged $(ls "$STAGE/videos" | wc -l) of $(wc -l < "$STAGE/videos.txt") videos ($(du -sh "$STAGE/videos" | cut -f1))"
fi

python -u scripts/eci/v0/v0_domains.py --domain "$DOMAIN" --video-dir "$VDIR" --stage-dir "$STAGE/tok" \
    --out-dir "$STAGE/out" --big-dir "/mnt/beegfs/locatgrp/shared/rcadei/eci_v0/${DOMAIN}${OUT_TAG}" \
    --num-workers 14 ${EXTRA_ARGS:-}

cp "$STAGE/out/"* "$RES/"
echo "copied results to $RES"
