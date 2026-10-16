#!/bin/bash
#
# NES + DINES on the online-model v4 layout (scripts/eci/online_v4.py) with the online ants page's NES settings
# (v36 antsfg; scripts/eci/ants30_chain.sh step 7, max pooling only): run_nes.py --primary-pooling max (core + pairs
# sets), run_nes_bouts.py --compare-pooling max (core + pairs), run_nes_latency.py --frame-pooling max (pairs);
# prefixes 128,256,1024, selection by |tau| (the runners' default), no foreground-count nuisance. Then DINES
# (scripts/eci/atlas_v37_methods.py, its 2026-10-16 settings) on the per-video outcomes of that result set.
#   LAY=eci_c1v4        --day: conditioned on the recording day (infested vs healthy: day F only, online_v4_nes.py)
#   LAY=eci_c1v4_noday  no --day: day-unmatched
# COMMITTED code only: HEAD is exported with `git archive` into the job's local dir (data / dataset / results
# symlinked back); the hash goes to results/vision/eci_online_v4/nes_code/<LAY>.json. The codes are staged to the
# local dir (--codes-root). CPU only.
#   LAY=eci_c1v4 sbatch --export=ALL scripts/eci/online_v4_nes.sh
#SBATCH --job-name=online_v4_nes
#SBATCH --output=results/vision/eci_online_v4/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
module load conda
conda activate crl
command -v git >/dev/null 2>&1 || module load git/2.35.1   # some CPU nodes (beta*) have no system git
command -v git >/dev/null 2>&1 || { echo 'no git: cannot snapshot the code'; exit 1; }
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
LAY=${LAY:?}
SAE=matryoshka_btk_1024_k16_antsfg_s0
export ANTS_ECI=$LAY
DAY=""; [ "$LAY" = "eci_c1v4" ] && DAY="--day"
HEAD=$(git rev-parse HEAD)
L=/localhome/$USER/${SLURM_JOB_ID:-local}_onlinev4
mkdir -p $L
trap 'rm -rf $L' EXIT
git archive HEAD src scripts/eci configs | tar -x -C $L
for d in data dataset results; do ln -s $REPO/$d $L/$d; done
mkdir -p $L/codes/$SAE
t0=$(date +%s)
cp dataset/ants/$LAY/codes/$SAE/{codes_max.npy,codes_mean.npy,n_fg.npy,config.json,DONE} $L/codes/$SAE/
echo "staged codes in $(( $(date +%s) - t0 ))s"
mkdir -p results/vision/eci_online_v4/nes_code
printf '{"head": "%s", "code_root": "%s", "sae": "%s", "ants_eci": "%s", "day": "%s", "host": "%s", "time": "%s"}\n' \
  "$HEAD" "$L" "$SAE" "$LAY" "$DAY" "$(hostname)" "$(date -Is)" > results/vision/eci_online_v4/nes_code/$LAY.json
echo "code: HEAD $HEAD exported to $L; ANTS_ECI $LAY; day '$DAY'; host $(hostname)"
W="python -u $L/scripts/eci/online_v4_nes.py"
CR="--codes-root $L/codes"
for set in core pairs; do
  echo "=== set $set: activation (per-video mean of codes_max)"
  $W run_nes.py --domain ants --sae $SAE $CR --primary-pooling max --analysis-set $set $DAY
  echo "=== set $set: bout rate"
  $W run_nes_bouts.py --domain ants --sae $SAE $CR --compare-pooling max --analysis-set $set $DAY
done
echo "=== set pairs: latency"
$W run_nes_latency.py --domain ants --sae $SAE $CR --frame-pooling max --analysis-set pairs $DAY
echo "=== DINES"
export ATLAS_V37_HEAD=$HEAD
$W atlas_v37_methods.py --res $L/results/vision/ants/$LAY/nes/$SAE --workers ${SLURM_CPUS_PER_TASK:-1} \
  --out-root $REPO/results/vision/ants/$LAY/methods
echo "done"
