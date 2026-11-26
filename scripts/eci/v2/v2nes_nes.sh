#!/bin/bash
#
# ECI v2-temporary NES: the v1 atlas NES runners with the v1 settings, on one v2 codes set (scripts/eci/v2/v2nes_codes.py),
# from a `git archive` of CODE_COMMIT (default HEAD at submission). CPU only.
# v1 settings (sanity.json of results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0 and
# results/vision/ants/eci30/nes/matryoshka_btk_1024_k16_antsfg_s0): codes_max primary pooling, per-video mean activation,
# t-test, Bonferroni 0.05, nuisance none, select tau, no recording-day covariate, core analysis set; ants layout eci30.
# Differences forced by the representation: --poolings max only (no mean-pool codes in v2), bouts --compare-pooling max,
# prefixes = the v2 Matryoshka levels 128,256,512,2048,4096 (v1: 128,256,1024 = its levels incl. the full dictionary).
#   RUN=mean|bouts|latency DOM=mice|ants SAE=v2tmp_PREDk16_s2901 sbatch --export=ALL scripts/eci/v2/v2nes_nes.sh
#   V1CHECK=1 RUN=mean DOM=mice SAE=matryoshka_btk_1024_k16_fg448al_s0 ... (the v1 code-path check: v1 args, v1 codes)
#SBATCH --job-name=v2nes_nes
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_nes/logs/%x_%j.out
#SBATCH --time=16:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=eta310,eta380
module load conda
conda activate crl
command -v git >/dev/null 2>&1 || module load git/2.35.1
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
RUN=${RUN:?}; DOM=${DOM:?}; SAE=${SAE:?}
CODE_COMMIT=${CODE_COMMIT:-$(git rev-parse HEAD)}
L=/localhome/$USER/${SLURM_JOB_ID:-local}_v2nes
mkdir -p $L
trap 'rm -rf $L' EXIT
git archive $CODE_COMMIT src scripts/eci configs | tar -x -C $L
for d in data dataset results; do ln -s $REPO/$d $L/$d; done
[ "$DOM" = "ants" ] && export ANTS_ECI=eci30
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
if [ "${V1CHECK:-0}" = "1" ]; then
  OUT=$REPO/results/vision/eci_v2_nes/v1check/$DOM
  [ "$DOM" = "mice" ] && CR=$REPO/dataset/mice/v1/eci/codes || CR=$REPO/dataset/ants/eci30/codes
  ARGS="--domain $DOM --sae $SAE --codes-root $CR --out-root $OUT --primary-pooling max --nuisance none --select tau"
else
  OUT=$REPO/results/vision/eci_v2_nes/$DOM/nes
  CR=/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_nes/$DOM/nes_codes
  ARGS="--domain $DOM --sae $SAE --codes-root $CR --out-root $OUT --nuisance none --select tau --prefixes 128,256,512,2048,4096"
fi
mkdir -p $OUT/code
printf '{"commit": "%s", "run": "%s", "dom": "%s", "sae": "%s", "v1check": "%s", "args": "%s", "host": "%s", "time": "%s"}\n' \
  "$CODE_COMMIT" "$RUN" "$DOM" "$SAE" "${V1CHECK:-0}" "$ARGS" "$(hostname)" "$(date -Is)" > $OUT/code/${RUN}_${SAE}_${SLURM_JOB_ID:-local}.json
echo "code $CODE_COMMIT -> $L; $RUN $DOM $SAE; $ARGS"
cd $L
case $RUN in
  mean)
    if [ "${V1CHECK:-0}" = "1" ]; then python -u scripts/eci/run_nes.py $ARGS
    else python -u scripts/eci/run_nes.py $ARGS --primary-pooling max --poolings max; fi ;;
  bouts)   python -u scripts/eci/run_nes_bouts.py $ARGS --compare-pooling max --frame-pooling max ;;
  latency) python -u scripts/eci/run_nes_latency.py $ARGS --frame-pooling max ;;
esac
echo "done $(date -Is)"
