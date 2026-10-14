#!/bin/bash
#
# PRIORS test (scripts/eci/priors.py): INV (D4 invariance) and PRED (1 s future-change prediction) loss terms on the
# C1 SAE, mice + ants. STEP = plan | encode | warmup | train | evaluate | sheets | clean (summary is login safe:
# python scripts/eci/priors.py summary). GPU: 'gpu' partition (never gpu100), excluding gpu242 / gpu150 / gpu230 /
# gpu241. Staging in /localhome/$USER/<job> (removed on exit). Outputs: results/vision/eci_priors/ (gitignored).
#
# Usage (from the repo root; mkdir -p results/vision/eci_priors/logs first):
#   STEP=plan sbatch -p defaultp --gres=none --cpus-per-task=4 --mem=64G --time=01:00:00 scripts/eci/priors.sh
#   STEP=encode EXTRA="--domain mice --workers 14" sbatch --time=01:00:00 scripts/eci/priors.sh
#   STEP=warmup EXTRA="--domain mice" sbatch --time=00:30:00 scripts/eci/priors.sh
#   STEP=train  EXTRA="--domain mice" sbatch --time=01:30:00 scripts/eci/priors.sh
#   STEP=evaluate EXTRA="--domain mice" sbatch --mem=200G --time=02:30:00 scripts/eci/priors.sh
#   STEP=sheets EXTRA="--browse-arm INV --workers 16" sbatch -p defaultp --gres=none --cpus-per-task=16 --mem=64G --time=01:00:00 scripts/eci/priors.sh
#
# Measured 2026-10-14 (results/vision/eci_priors/summary.json, tables.md; 3 SAE seeds; cf = cross-fitted best neuron;
# ctl r = count-controlled held-out per-video mean vs the annotated rate). Checks: C1r = res_pool c448_s0 bit for bit;
# C1 / C1dep cf metrics reproduce res_pool exactly (max abs diff 0); decoded JPGs byte-identical (228/228 videos);
# re-encoded tokens cos >= 0.9991 vs the store.
#   lambda / mu (25% of L_rec after 1000 warm-up steps): mice 791.4 / 596.8, ants 639.9 / 510.9.
#   INV collapses into a dense always-on component: 90% of pre-activations > 0 (C1 14%), raw L_inv 0.65 -> 0.018,
#     FVE 0.780 -> 0.746 (mice); 4045 / 4096 latents eligible. The term is met without making the codes invariant.
#                        cf AUROC at 4096  C1 / INV / PRED / INVPRED          honest top-1% (C1 / best arm)
#   mice nose_nose       0.700 / 0.682 / 0.677 / 0.689                        0.117 / 0.119 (INV)
#   mice nose_tail       0.753 / 0.726 / 0.709 / 0.741                        0.096 / 0.089
#   ants groom_any       0.883 / 0.861 / 0.854 / 0.832
#   ants onlid_yellow    0.963 / 0.969 / 0.960 / 0.874
#   PRIMARY (mice honest top-1% >= 0.40): NOT met (best 0.119). KEEP: neither INV nor PRED nor INVPRED is kept at any
#   prefix (every arm drops > 0.02 cf AUROC on >= 1 ants label, groom_any -0.022 / -0.029 / -0.051; in mice only
#   INVPRED at 4096 has no drop). WIDTH (C1): 512 / 1024 match 4096 within
#   0.003 on 5 of 6 labels; ants onlid_yellow drops 0.098 / 0.085. Compute: about 1.5 GPU-h (L40S) + 1 CPU-h.
#SBATCH --job-name=eci_priors
#SBATCH --output=results/vision/eci_priors/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=gpu242,gpu150,gpu230,gpu241
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_pr
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP extra ${EXTRA:-}"
df -h /localhome | tail -1
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/priors.py $STEP ${EXTRA:-}
