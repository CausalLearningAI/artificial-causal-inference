#!/bin/bash
#
# Sleeper (CPU, 1 core) for the atlas v37 R4 real heat maps. For each key in KEYS it waits until the v37 builder's clip
# picks exist, its peak-patch step has written a patch file for every picked neuron, and nothing in patch/ changed for
# 3 min; then it runs scripts/eci/atlas_v37_r4heat_key.sh (GPU parts per encode GPU + assemble with MOVE=1). If the
# submitter fails (e.g. a clip video whose encode log is not there yet), it retries on the next poll. Polls every 2 min.
#   KEYS="mice ants_v4" sbatch scripts/eci/atlas_v37_r4heat_sleeper.sh
#SBATCH --job-name=r4heat_sleep
#SBATCH --output=results/vision/eci_atlas_v37/logs/%x_%j.out
#SBATCH --partition=defaultp
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
module load conda
conda activate crl
set -uo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
todo="$KEYS"
while [ -n "$todo" ]; do
  left=""
  for k in $todo; do
    st=$(python - "$k" <<'PY'
import json, sys, time
from pathlib import Path
k = sys.argv[1]
name = json.loads(Path(f'results/vision/eci_rollout_r4/{k}/sae/best.json').read_text())['d2']['name']
w = Path(f'results/vision/eci_atlas_v37/res/{k}_d2/{name}_fnorm/_cache/explorer')
try:
    p = json.loads((w / 'picks.json').read_text())
except Exception:
    print('nopicks'); sys.exit()
js = list(p['by_key'].get('max', {}))
fs = [w / 'patch' / f'x{int(j):04d}.npz' for j in js]
have = [f for f in fs if f.exists()]
if not js or len(have) < len(js):
    print(f'patch {len(have)}/{len(js)}'); sys.exit()
age = time.time() - max(f.stat().st_mtime for f in have + [w / 'picks.json'])
print('ready' if age > 180 else f'settling {age:.0f}s')
PY
)
    echo "$(date +%T) $k: $st"
    if [ "$st" = "ready" ] && MOVE=1 bash scripts/eci/atlas_v37_r4heat_key.sh $k; then
      echo "$(date +%T) $k: submitted"
    else
      left="$left $k"
    fi
  done
  todo=$(echo $left)
  [ -n "$todo" ] && sleep 120
done
echo "$(date +%T) all submitted"
