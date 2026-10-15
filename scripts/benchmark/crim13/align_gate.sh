#!/bin/bash
#
# CRIM13 alignment - post-encode chain (CPU job on defaultp, runs after the 4 encode jobs; rules: eci_align.sh).
# 1. mask recall of the BC (pale) intruders from the token-store frame tables (own cell) -> OUT/res/mask_gate.json
# 2. addendum 1: if that recall < 0.90, submit the EXPLORATORY C4 arm (train on random cells, then all-cell read-out)
# 3. sheets (CPU) after C1; score (CPU) after C1 C2 C3 probe [C4]; cleanup of the big token arrays (*.npy except
#    pos40 / rndpos40) after C1 C2 C3 probe [C4 train] succeeded.
# usage: sbatch ... align_gate.sh  with env C1 C2 C3 PR = job ids of the train / probe jobs
#SBATCH --job-name=cr_gate
#SBATCH --output=results/vision/benchmark_crim13/align/logs/%x_%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
module load conda
conda activate crl
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
L=results/vision/benchmark_crim13/align/logs
mkdir -p results/vision/benchmark_crim13/align/res
REC=$(python -I - <<'EOF'
import json, sys
sys.path.insert(0, 'scripts/benchmark/crim13')
import pandas as pd
from align_common import OUT, PLAN, TOK
F = pd.concat([pd.read_parquet(j) for j in sorted(TOK.glob('*of*_frames.parquet')) if 'smoke' not in j.name])
F = F.merge(pd.read_parquet(PLAN / 'videos.parquet')[['video', 'strain']], on='video')
r = {s: float(F[F.has_int & (F.strain == s)].rec_int.mean()) for s in ('BC', 'C57')}
r['resident'] = float(F[F.has_res].rec_res.mean())
r['frames'] = len(F)
(OUT / 'res' / 'mask_gate.json').write_text(json.dumps(r, indent=1))
print(r['BC'])
EOF
)
echo "BC-intruder own-cell recall $REC"
DEPS="$C1:$C2:$C3:$PR"
SC=""
if python -c "import sys; sys.exit(0 if float('$REC') < 0.90 else 1)"; then
  echo "BC recall < 0.90 -> EXPLORATORY C4 (addendum 1)"
  X="--exclude=gpu242,gpu150,gpu230,gpu241,gpu240"
  T4=$(sbatch --parsable -p gpu --gres=gpu:1 $X --constraint='L40S|A100|A40|A10' --cpus-per-task=8 --mem=160G \
       --time=01:30:00 --job-name=cr_tr_C4 --output=$L/%x_%j.out --export=ALL,PY=align_train,EXTRA="--arm C4" \
       scripts/benchmark/crim13/align_job.sh)
  E4=$(sbatch --parsable -p gpu --gres=gpu:1 $X --constraint='L40S|A100|A40|A10' --cpus-per-task=16 --mem=110G \
       --time=02:00:00 --job-name=cr_c4_eval --output=$L/%x_%j.out --dependency=afterok:$T4 \
       --export=ALL,PY=align_c4,EXTRA="" scripts/benchmark/crim13/align_job.sh)
  sbatch --parsable -p defaultp --cpus-per-task=4 --mem=16G --time=00:30:00 --job-name=cr_sheets_C4 \
    --output=$L/%x_%j.out --dependency=afterok:$E4 --export=ALL,PY=align_sheets,EXTRA="C4" scripts/benchmark/crim13/align_job.sh
  DEPS="$DEPS:$T4"
  SC=":$E4"
  echo "C4 train $T4 eval $E4"
fi
S=$(sbatch --parsable -p defaultp --cpus-per-task=4 --mem=16G --time=00:30:00 --job-name=cr_sheets \
    --output=$L/%x_%j.out --dependency=afterok:$C1 --export=ALL,PY=align_sheets,EXTRA="mask C1" scripts/benchmark/crim13/align_job.sh)
R=$(sbatch --parsable -p defaultp --cpus-per-task=2 --mem=8G --time=00:15:00 --job-name=cr_score \
    --output=$L/%x_%j.out --dependency=afterany:$C1:$C2:$C3:$PR:$S$SC --export=ALL,PY=align_score,EXTRA="" \
    scripts/benchmark/crim13/align_job.sh)
CL=$(sbatch --parsable -p defaultp --cpus-per-task=1 --mem=2G --time=00:15:00 --job-name=cr_cleanup \
    --output=$L/%x_%j.out --dependency=afterok:$DEPS \
    --wrap "cd /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/crim13 && ls -la && find . -maxdepth 1 -name '*.npy' ! -name '*pos40.npy' -print -delete && du -sh .")
echo "sheets $S score $R cleanup $CL"
