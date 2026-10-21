#!/bin/bash
#
# ECI v2 mice extra families: NES (3 runners: mean / bout rate / latency, max pooling) + NESS on one codes set with the
# in-memory MiceDomain patch of scripts/eci/v2/nes_mice_dispatch.py, from a git archive of CODE_COMMIT. CPU only.
#   MODE=S SAE=v2_mice_c0_w2048_s7_smooth sbatch --export=ALL scripts/eci/v2/nes_mice.sh
#   MODE=strata:ash1l:all SAE=... sbatch --export=ALL scripts/eci/v2/nes_mice.sh
#SBATCH --job-name=v2_nesmice
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2/logs/%x_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=eta295,eta296,eta297,eta298
module load conda
conda activate crl
command -v git >/dev/null 2>&1 || module load git/2.35.1
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
MODE=${MODE:?}; SAE=${SAE:?}; PREFIXES=${PREFIXES:-128,512,2048}
CODE_COMMIT=${CODE_COMMIT:-$(git rev-parse HEAD)}
L=/localhome/$USER/${SLURM_JOB_ID:-local}_v2nesmice
mkdir -p $L 2>/dev/null || { L=/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2/tmp/$(basename $L); mkdir -p $L; }  # nodes without /localhome
trap 'rm -rf $L' EXIT
git archive $CODE_COMMIT src scripts/eci configs | tar -x -C $L
for d in data dataset results; do ln -s $REPO/$d $L/$d; done
export NES_CODE_ROOT=$L V2_MICE_MODE=$MODE OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
W="python -u $L/scripts/eci/v2/nes_mice_dispatch.py"
RES=results/vision/mice/eci/nes/$SAE
if [ "$MODE" = "S" ]; then
  SUB=family_S; EXT=""; OUTN=$REPO/results/vision/eci_v2/ness_S
  AIDS=$(python -c "print(','.join([f'S_stage{s}' for s in range(1,7)]+[f'S_{g}_stage{s}' for g in ('het','wt') for s in range(1,7)]))")
else
  IFS=: read _ LN SX <<< "$MODE"; SUB=subsets/${LN}_${SX}; EXT="--line $LN --sex $SX"; OUTN=$REPO/results/vision/eci_v2/ness_strata/${LN}_${SX}
  AIDS=$(python -c "print(','.join([f'A_{g}_{t}' for g in ('het','wt') for t in ('1to2','2to3','4to5','5to6')]+[f'B_stage{s}' for s in range(1,7)]))")
fi
mkdir -p results/vision/eci_v2/nes_code
printf '{"commit": "%s", "sae": "%s", "mode": "%s", "subdir": "%s", "time": "%s"}\n' "$CODE_COMMIT" "$SAE" "$MODE" "$SUB" "$(date -Is)" \
  > results/vision/eci_v2/nes_code/mice_${MODE//:/_}_${SAE}.json
COMMON="--domain mice --sae $SAE --prefixes $PREFIXES --nuisance nfg --select p --subdir $SUB"
echo "=== $MODE $SAE: mean";    $W run_nes.py $COMMON $EXT --primary-pooling max --poolings max --skip-frame
echo "=== $MODE $SAE: bouts";   $W run_nes_bouts.py $COMMON $EXT --compare-pooling max
echo "=== $MODE $SAE: latency"; $W run_nes_latency.py $COMMON --frame-pooling max
echo "=== $MODE $SAE: NESS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ATLAS_V37_HEAD=$CODE_COMMIT
$W atlas_v37_methods.py --res $L/$RES --nes-subdir $SUB --analyses $AIDS --no-family-c --workers ${SLURM_CPUS_PER_TASK:-1} --out-root $OUTN
echo done
