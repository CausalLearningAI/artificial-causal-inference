#!/bin/bash
#
# Ants v3 t=2 vs t=8 on an ECI v2 codes set (layout eci_r4v3p; scripts/eci/v2 copy of scripts/eci/r4_v3_2v8.sh), NES and
# DINES (= NESS), with the settings of the R4 NES runs (scripts/eci/rollout_r4_nes.sh: codes_max, mean / bout rate /
# latency, --select p, --nuisance nfg, prefixes 128,256,512, Bonferroni) EXCEPT the recording-day covariate: t=8 was
# filmed only on day C and t=2 only on days A/B, so the --day runs skip this pair (no shared day). Searched here
# without --day (n_fg only), as the runner's pairs_noday sensitivity run. t=2 vs t=6 is not re-run (the page keeps the
# published --day results). Committed code only (git archive of HEAD); codes and the NES caches staged to local disk;
# results copied back once:
#   NES   -> results/vision/ants/eci_r4v3p/nes/<sae>/v3_2_vs_8_noday/{.,maxpool_bouts,maxpool_latency}
#   DINES -> results/vision/eci_v2/ness_v3_2v8/ants_eci_r4v3p/<sae>/dines
#   sbatch scripts/eci/r4_v3_2v8.sh
#SBATCH --job-name=v2_v3_2v8
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2/logs/%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
module load conda
conda activate crl
command -v git >/dev/null 2>&1 || module load git/2.35.1
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
SAE=${SAE:?}; CODE_COMMIT=${CODE_COMMIT:-$(git rev-parse HEAD)}
SUB=v3_2_vs_8_noday
NES=results/vision/ants/eci_r4v3p/nes
L=/localhome/$USER/${SLURM_JOB_ID:-local}_v3_2v8
mkdir -p $L 2>/dev/null || L=/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2/tmp/$(basename $L)  # nodes without /localhome
mkdir -p $L/code $L/codes/$SAE $L/nes/$SAE $L/methods
trap 'rm -rf $L' EXIT
HEAD=$CODE_COMMIT
git archive $HEAD src scripts/eci configs | tar -x -C $L/code
for d in data dataset results; do ln -s $REPO/$d $L/code/$d; done
cp -L dataset/ants/eci_r4v3p/codes/$SAE/* $L/codes/$SAE/
cp -rL $NES/$SAE/_cache $L/nes/$SAE/
echo "HEAD $HEAD; staged $(du -sh $L | cut -f1) to $L; host $(hostname)"
export ANTS_ECI=eci_r4v3p NES_CODE_ROOT=$L/code ATLAS_V37_HEAD=$HEAD
cd $L/code
W="python -u scripts/eci/rollout_r4_nes.py"; [ "${V2_ZERO_THR:-0}" = "1" ] && W="python -u scripts/eci/v2/nes_v2_dispatch.py scripts/eci/rollout_r4_nes.py"
COMMON="--domain ants --sae $SAE --prefixes 128,512,2048 --nuisance nfg --select p --analysis-set core --subdir $SUB --codes-root $L/codes --out-root $L/nes"
echo "=== NES activation (per-video mean of codes_max)"
$W run_nes.py $COMMON --primary-pooling max --poolings max --skip-frame
echo "=== NES bout rate"
$W run_nes_bouts.py $COMMON --compare-pooling max
echo "=== NES latency"
$W run_nes_latency.py $COMMON --frame-pooling max
mkdir -p $REPO/$NES/$SAE/$SUB
cp -r $L/nes/$SAE/$SUB/. $REPO/$NES/$SAE/$SUB/
echo "=== DINES (NESS) on the same per-video outcomes, NES settings of $SUB (no day)"
python -u scripts/eci/atlas_v37_methods.py --res $L/code/$NES/$SAE --nes-subdir $SUB --analyses v3_2_vs_8 \
  --out-root $L/methods --workers ${SLURM_CPUS_PER_TASK:-1}
mkdir -p $REPO/results/vision/eci_v2/ness_v3_2v8
cp -r $L/methods/. $REPO/results/vision/eci_v2/ness_v3_2v8/
echo "done"
