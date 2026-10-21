#!/bin/bash
#
# ECI v2: NES (the existing runners, three temporal aggregations, max pooling) and NESS (atlas_v37_methods.py =
# src/eci/dines.py driver) on one v2 codes set, from a `git archive` of the pinned commit CODE_COMMIT (default HEAD at
# submission; recorded with the sha1 of src/eci/nes.py / dines.py in results/vision/eci_v2/nes_code/). CPU only.
# Dispatchers (unchanged, from the archive): scripts/eci/rollout_r4_nes.py (mice / ants / frogs juveniles; ANTS_ECI
# patches the ants null analyses) and scripts/eci/tadpole_r4_nes.py (tadpoles: TADPOLE_TAG=_r4, WT vs FoxP1 / En1, QC
# exclusion, NESS parse_res patch).
#   KIND=mice SAE=v2_mice_... PREFIXES=128,512,2048 sbatch --export=ALL scripts/eci/v2/nes.sh
#   KIND=ants ANTS_ECI=eci_r4v3p SAE=... ; KIND=frogs SAE=... ; KIND=tadpoles SAE=...
#SBATCH --job-name=v2_nes
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2/logs/%x_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
module load conda
conda activate crl
command -v git >/dev/null 2>&1 || module load git/2.35.1
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
KIND=${KIND:?}; SAE=${SAE:?}; PREFIXES=${PREFIXES:-128,512,2048}
CODE_COMMIT=${CODE_COMMIT:-$(git rev-parse HEAD)}
L=/localhome/$USER/${SLURM_JOB_ID:-local}_v2nes
mkdir -p $L 2>/dev/null || { L=/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2/tmp/$(basename $L); mkdir -p $L; }  # nodes without /localhome
trap 'rm -rf $L' EXIT
git archive $CODE_COMMIT src scripts/eci configs | tar -x -C $L
for d in data dataset results; do ln -s $REPO/$d $L/$d; done
export NES_CODE_ROOT=$L
mkdir -p results/vision/eci_v2/nes_code results/vision/eci_v2/ness
H=$(cd $L && sha1sum src/eci/nes.py src/eci/dines.py scripts/eci/atlas_v37_methods.py scripts/eci/run_nes.py | awk '{printf "%s:%s ", $2, substr($1,1,12)}')
NC=$(git log -1 --format=%h $CODE_COMMIT -- src/eci/nes.py src/eci/dines.py)
TAG=${KIND}${ANTS_ECI:+_$ANTS_ECI}
printf '{"commit": "%s", "nes_dines_last_change_commit": "%s", "file_sha1": "%s", "sae": "%s", "kind": "%s", "ants_eci": "%s", "prefixes": "%s", "time": "%s"}\n' \
  "$CODE_COMMIT" "$NC" "$H" "$SAE" "$KIND" "${ANTS_ECI:-}" "$PREFIXES" "$(date -Is)" > results/vision/eci_v2/nes_code/${TAG}_${SAE}.json
echo "code $CODE_COMMIT (nes/dines last changed in $NC) -> $L; $H"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ "$KIND" = "tadpoles" ]; then
  W="python -u $L/scripts/eci/tadpole_r4_nes.py"; DOM=tadpoles; RES=results/vision/frogs/eci_tadpole_r4/nes/$SAE
else
  W="python -u $L/scripts/eci/rollout_r4_nes.py"; DOM=$KIND
  case $KIND in mice) RES=results/vision/mice/eci/nes/$SAE ;; frogs) RES=results/vision/frogs/eci/nes/$SAE ;;
    ants) RES=results/vision/ants/$ANTS_ECI/nes/$SAE ;; esac
fi
COMMON="--domain $DOM --sae $SAE --prefixes $PREFIXES --nuisance nfg --select p"
DAY=""; [ "$KIND" = "ants" ] && DAY="--day"
SETS="core"; [ "$KIND" = "ants" ] && SETS="core pairs"
for set in $SETS; do
  echo "=== $SAE set $set: per-video mean of codes_max";  $W run_nes.py $COMMON $DAY --analysis-set $set --primary-pooling max --poolings max
  echo "=== $SAE set $set: bout rate";  $W run_nes_bouts.py $COMMON $DAY --analysis-set $set --compare-pooling max
  echo "=== $SAE set $set: latency";  $W run_nes_latency.py $COMMON $DAY --analysis-set $set --frame-pooling max
done
if [ "$KIND" = "mice" ]; then
  echo "=== camera-period adjusted family B (sensitivity)"; $W run_nes.py $COMMON --primary-pooling max --poolings max --period adjust --primary-only
fi
if [ "$KIND" = "ants" ]; then
  echo "=== without the day covariate (sensitivity; day-confounded pairs such as v3 2 vs 8 are testable only here)"
  $W run_nes.py $COMMON --analysis-set pairs --primary-pooling max --poolings max --subdir pairs_noday --primary-only
  $W run_nes_bouts.py $COMMON --analysis-set pairs --compare-pooling max --subdir pairs_noday --primary-only
  $W run_nes_latency.py $COMMON --analysis-set pairs --frame-pooling max --subdir pairs_noday --primary-only || echo "latency noday: not supported"
fi
echo "=== NESS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ATLAS_V37_HEAD=$CODE_COMMIT
if [ "$KIND" = "tadpoles" ]; then
  $W atlas_v37_methods.py --res $L/$RES --workers ${SLURM_CPUS_PER_TASK:-1} --out-root $REPO/results/vision/eci_v2/ness
else
  python -u $L/scripts/eci/atlas_v37_methods.py --res $L/$RES --workers ${SLURM_CPUS_PER_TASK:-1} --out-root $REPO/results/vision/eci_v2/ness
  if [ "$KIND" = "ants" ] && [ -f $RES/pairs_noday/summary.csv ]; then
    python -u $L/scripts/eci/atlas_v37_methods.py --res $L/$RES --nes-subdir pairs_noday --workers ${SLURM_CPUS_PER_TASK:-1} \
      --out-root $REPO/results/vision/eci_v2/ness_noday || echo "NESS noday failed"
  fi
fi
echo done
