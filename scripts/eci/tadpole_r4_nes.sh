#!/bin/bash
#
# NES + NESS on the tadpole r4 codes (scripts/eci/tadpole_r4_nes.py), the r4 juvenile frogs settings
# (scripts/eci/rollout_r4_nes.sh, DOMAIN=frogs): codes_max, per-video mean (run_nes.py), bout rate (run_nes_bouts.py),
# latency (run_nes_latency.py); --select p, --nuisance nfg, no session / day covariate (as the juveniles: every session
# holds one group, so group is not separable from session); prefixes 128,256,512. Then NESS (= DINES, src/eci/dines.py,
# scripts/eci/atlas_v37_methods.py) on the same result set. COMMITTED code only: CODE_COMMIT (pinned at submit time)
# is exported with `git archive` into the job's local dir; the commit and the hashes of the NES / NESS files are recorded
# in results/vision/eci_rollout_r4/nes_code/frogs_tadpole_<sae>.json. CPU only.
#   SAE=matryoshka_btk_512_k16_r4d2_s0_fnorm POOLINGS=max CODE_COMMIT=<sha> sbatch --export=ALL scripts/eci/tadpole_r4_nes.sh
#SBATCH --job-name=tad_nes
#SBATCH --output=results/vision/eci_rollout_r4/logs/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
module load conda
conda activate crl
command -v git >/dev/null 2>&1 || module load git/2.35.1
command -v git >/dev/null 2>&1 || { echo 'no git'; exit 1; }
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 TADPOLE_TAG=_r4
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
SAE=${SAE:?}
CODE_COMMIT=${CODE_COMMIT:?}
PREFIXES=${PREFIXES:-128,256,512}
PL=""; [ -n "${POOLINGS:-}" ] && PL="--poolings $POOLINGS"
L=/localhome/$USER/${SLURM_JOB_ID:-local}_tadnes
mkdir -p $L
trap 'rm -rf $L' EXIT
git archive $CODE_COMMIT src scripts/eci configs | tar -x -C $L
for d in data dataset results; do ln -s $REPO/$d $L/$d; done
export NES_CODE_ROOT=$L
mkdir -p results/vision/eci_rollout_r4/nes_code
H=$(cd $L && sha1sum src/eci/dines.py src/eci/nes.py scripts/eci/atlas_v37_methods.py scripts/eci/run_nes.py 2>/dev/null | awk '{printf "%s:%s ", $2, substr($1,1,12)}' || true)
NESSC=$(git log -1 --format=%h $CODE_COMMIT -- src/eci/dines.py scripts/eci/atlas_v37_methods.py)
printf '{"commit": "%s", "ness_last_change_commit": "%s", "file_sha1": "%s", "code_root": "%s", "sae": "%s", "domain": "tadpoles", "TADPOLE_TAG": "_r4", "time": "%s"}\n' \
  "$CODE_COMMIT" "$NESSC" "$H" "$L" "$SAE" "$(date -Is)" > results/vision/eci_rollout_r4/nes_code/frogs_tadpole_${SAE}.json
python - <<PY
import json; f='results/vision/eci_rollout_r4/nes_code/frogs_tadpole_${SAE}.json'; j=json.load(open(f))
j['qc'] = json.load(open('results/vision/eci_rollout_r4/frogs_tadpole/qc.json')); json.dump(j, open(f, 'w'), indent=1)
PY
echo "code $CODE_COMMIT (NESS files last changed in $NESSC) -> $L"
W="python -u $L/scripts/eci/tadpole_r4_nes.py"
COMMON="--domain tadpoles --sae $SAE --prefixes $PREFIXES --nuisance nfg --select p"
echo "=== $SAE: activation (per-video mean of codes_max)"
$W run_nes.py $COMMON --analysis-set core --primary-pooling max $PL
echo "=== $SAE: bout rate"
$W run_nes_bouts.py $COMMON --analysis-set core --compare-pooling max
echo "=== $SAE: latency"
$W run_nes_latency.py $COMMON --analysis-set core --frame-pooling max
echo "=== $SAE: NESS (DINES)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ATLAS_V37_HEAD=$CODE_COMMIT
$W atlas_v37_methods.py --res $L/results/vision/frogs/eci_tadpole_r4/nes/$SAE --workers ${SLURM_CPUS_PER_TASK:-1} \
  --out-root $REPO/results/vision/eci_rollout_r4/ness
echo done
