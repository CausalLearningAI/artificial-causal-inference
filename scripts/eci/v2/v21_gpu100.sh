#!/bin/bash
# NOT RUN. Switch the pending v2.1 store jobs to the gpu100 partition (H100), only if the user approves.
# Cancels the pending checks / arrays / manifests (they have not started; completed videos would be skipped anyway)
# and resubmits them identically on -p gpu100 (no --constraint; same exclusions are irrelevant there).
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
scancel 1323884 1323885 1323886 1323887 1323888 1323889 1323891 1323892 1323893
declare -A NT=([ants]=58 [mice]=162 [frogs]=51)
for d in ants mice frogs; do
  c=$(sbatch --parsable -J v21chk_$d -p gpu100 --gres=gpu:1 -c 8 --mem=64G --time=2:00:00 --export=ALL,EXTRA="check --domain $d" scripts/eci/v2/v21_store.sh)
  a=$(sbatch --parsable -J v21enc_$d -p gpu100 --gres=gpu:1 -c 8 --mem=64G --time=12:00:00 --array=0-$((NT[$d]-1)) --dependency=afterok:$c --export=ALL,EXTRA="encode --domain $d --ntask ${NT[$d]}" scripts/eci/v2/v21_store.sh)
  m=$(sbatch --parsable -J v21man_$d -p defaultp -c 4 --mem=16G --time=6:00:00 --dependency=afterany:$a --export=ALL,EXTRA="manifest --domain $d" scripts/eci/v2/v21_store.sh)
  echo "$d check $c array $a manifest $m"
done
