#!/bin/bash
#
# NES on R4 codes with the C1-rollout settings (scripts/eci/rollout_c1_nes.sh): codes_max, per-video mean (run_nes.py),
# bout rate (run_nes_bouts.py), latency (run_nes_latency.py); --select p, --nuisance nfg, --day for ants; prefixes
# 128,256,512 (512-latent SAEs). The NES code is this repo's src/eci + scripts/eci runners. Its commit is recorded; when
# the NES files have uncommitted changes (another session edits them), HEAD is exported with `git archive` into the
# job's local dir and run from there (data / dataset / results symlinked back), so only committed code runs. CPU only.
#   DOMAIN=mice SAE=matryoshka_btk_512_k16_r4d2_s0 sbatch --export=ALL scripts/eci/rollout_r4_nes.sh
#   DOMAIN=ants ANTS_ECI=eci_r4v3 SAE=... sbatch --export=ALL scripts/eci/rollout_r4_nes.sh
#SBATCH --job-name=r4_nes
#SBATCH --output=results/vision/eci_rollout_r4/logs/%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
DOMAIN=${DOMAIN:?}
SAE=${SAE:?}
PREFIXES=${PREFIXES:-128,256,512}
TAG=${TAG:-$DOMAIN}
NESFILES="src/eci scripts/eci/run_nes.py scripts/eci/run_nes_bouts.py scripts/eci/run_nes_latency.py"
HEAD=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain -- $NESFILES | wc -l)
export NES_CODE_ROOT=$REPO
if [ "$DIRTY" != "0" ]; then
  L=/localhome/$USER/${SLURM_JOB_ID:-local}_r4nes
  mkdir -p $L
  trap 'rm -rf $L' EXIT
  git archive HEAD src scripts/eci configs | tar -x -C $L
  for d in data dataset results; do ln -s $REPO/$d $L/$d; done
  export NES_CODE_ROOT=$L
fi
mkdir -p results/vision/eci_rollout_r4/nes_code
printf '{"head": "%s", "dirty_nes_files": %s, "code_root": "%s", "sae": "%s", "domain": "%s", "ants_eci": "%s", "time": "%s"}\n' \
  "$HEAD" "$DIRTY" "$NES_CODE_ROOT" "$SAE" "$DOMAIN" "${ANTS_ECI:-}" "$(date -Is)" > results/vision/eci_rollout_r4/nes_code/${TAG}_${SAE}.json
git status --porcelain -- $NESFILES || true
echo "NES code: HEAD $HEAD, dirty NES files $DIRTY, running from $NES_CODE_ROOT"
W="python -u scripts/eci/rollout_r4_nes.py"
COMMON="--domain $DOMAIN --sae $SAE --prefixes $PREFIXES --nuisance nfg --select p"
DAY=""; [ "$DOMAIN" = "ants" ] && DAY="--day"
SETS="core"; [ "$DOMAIN" = "ants" ] && SETS="core pairs"
for set in $SETS; do
    echo "=== $SAE set $set: activation (per-video mean of codes_max)"
    $W run_nes.py $COMMON $DAY --analysis-set $set --primary-pooling max
    echo "=== $SAE set $set: bout rate"
    $W run_nes_bouts.py $COMMON $DAY --analysis-set $set --compare-pooling max
    echo "=== $SAE set $set: latency"
    $W run_nes_latency.py $COMMON $DAY --analysis-set $set --frame-pooling max
done
if [ "$DOMAIN" = "mice" ]; then
    echo "=== $SAE: camera-period adjusted family B (sensitivity)"
    $W run_nes.py $COMMON --primary-pooling max --period adjust --primary-only
fi
if [ "$DOMAIN" = "ants" ]; then
    echo "=== $SAE: without the day covariate (sensitivity; the day-confounded pairs, e.g. v3 t=2 vs t=8, are testable only here)"
    $W run_nes.py $COMMON --analysis-set pairs --primary-pooling max --subdir pairs_noday --primary-only
    $W run_nes_bouts.py $COMMON --analysis-set pairs --compare-pooling max --subdir pairs_noday --primary-only
fi
echo "done"
