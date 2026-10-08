#!/bin/bash
#
# Per-mouse / per-pair token checks, mice v1 (scripts/eci/mice_pairs_*.py). STEP = select | sam | score | probe,
# SET = check1 | bsub. Outputs in results/vision/eci_mice_pairs/<set>/ (gitignored).
#   select  CPU   targets, +-20 s anchor search, targets.tar / clips.tar; SET=both does both sets (mice_pairs_select.py)
#   sam     GPU   pre-registered combined splitter (propagation else B1npk)     (mice_pairs_sam.py)
#   score   CPU   automatic rule, per-branch table, sheets; bsub: mask geometry (mice_pairs_score.py)
#   probe   GPU   bsub only: pair-token probes P1 / P2 / ablations vs patch mil_ctx (mice_pairs_probe.py)
# SAM 2 from tools/sam2 (+ tools/sam2_deps) on PYTHONPATH, as scripts/eci/split_test.sh. Inputs staged to
# /localhome/$USER/$SLURM_JOB_ID and removed at the end.
#
# Measured 2026-10-08 (results/vision/eci_mice_pairs/<set>/table.json, check1/visual_check.json, bsub/probe/results.json):
#   Check 1 (fresh 1000 contact + 1000 non-contact frames): contact automatic 89.2% (propagation branch 95.7% on 85.8%
#     coverage, B1npk fallback 50.0%), visual 46/50 (Claude, not blind to the auto verdicts); non-contact automatic
#     80.1% (propagation 93.3% on 65.8%, fallback 54.7%), visual 20/25. Gate (>= 85% contact, both checks): PASS.
#   Check 2 (B subset, 16,775 / 20,000 frames pass the automatic rule): mean over 5 pool folds x 3 seeds, AUROC / AP
#     nose_nose  P1 pair MIL 0.879 / 0.283   P1 no contact 0.852 / 0.221   P1 geometry 0.691 / 0.052   P2 linear max
#                0.874 / 0.280   P3 patch mil_ctx 0.879 / 0.268   P3_all 0.879 / 0.260
#     nose_tail  P1 0.912 / 0.252   no contact 0.894 / 0.208   geometry 0.758 / 0.053   P2 0.885 / 0.201
#                P3 0.916 / 0.244   P3_all 0.918 / 0.248
#   "pair tokens help" (P1 vs P3: AP >= 1.3x or AUROC >= +0.03): NO (AP x1.06 / x1.03, AUROC +0.000 / -0.004).
#   "a single pair-level direction suffices" (P2 >= 80% of P1's AP): nose_nose yes (99%), nose_tail no (79.8%).
#
# Usage:
#   SET=both STEP=select sbatch --export=ALL --partition=defaultp --gres=none --time=01:30:00 --cpus-per-task=16 --mem=96G scripts/eci/mice_pairs.sh
#   SET=bsub STEP=sam EXTRA="--shard 0 --n-shards 4" sbatch --export=ALL --gres=gpu:L40S:1 --mem=80G --time=01:10:00 scripts/eci/mice_pairs.sh
#   SET=check1 STEP=sam sbatch --export=ALL --time=01:00:00 scripts/eci/mice_pairs.sh
#SBATCH --job-name=mice_pairs
#SBATCH --output=logs/mice_pairs_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
export STAGE=/localhome/$USER/${SLURM_JOB_ID:-local}
mkdir -p $STAGE
trap 'rm -rf $STAGE' EXIT
T=/nfs/scistore19/locatgrp/rcadei/tools
export PYTHONPATH=$T/sam2:$T/sam2_deps:${PYTHONPATH:-}
D=results/vision/eci_mice_pairs/$SET
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP set $SET"
df -h /localhome | tail -1
case $STEP in
  select)
    python -u scripts/eci/mice_pairs_select.py --set $SET --workers ${SLURM_CPUS_PER_TASK:-8} ${EXTRA:-} ;;
  sam)
    nvidia-smi -L
    cp $D/targets.tar $D/clips.tar $STAGE/
    python -u scripts/eci/mice_pairs_sam.py --set $SET ${EXTRA:-} ;;
  score)
    cp $D/targets.tar $STAGE/
    python -u scripts/eci/mice_pairs_score.py --set $SET --workers ${SLURM_CPUS_PER_TASK:-8} ${EXTRA:-} ;;
  probe)
    nvidia-smi -L
    B=results/vision/eci_repr_diag/mice/b
    cp $B/tok448.npy $B/pos448.i16 $B/frames.npz $B/subset.parquet $STAGE/
    python -u scripts/eci/mice_pairs_probe.py ${EXTRA:-} ;;
esac
