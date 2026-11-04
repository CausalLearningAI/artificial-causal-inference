#!/bin/bash
# v2.4: cancel a training job as soon as all its SAE checkpoints are saved (codes then run on CPU, saving H100 time).
# usage: sbatch v24_stopper.sh JOBID:NSAE ...
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310,eta380
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_4/logs/%x_%j.out
L=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_4/logs
left=("$@")
while [ ${#left[@]} -gt 0 ]; do
  nxt=()
  for jn in "${left[@]}"; do
    j=${jn%%:*}; n=${jn##*:}
    f=$(ls $L/*_$j.out 2>/dev/null | head -1)
    c=$(grep -c "eval FVE" "$f" 2>/dev/null); c=${c:-0}
    st=$(squeue -h -j $j -o %T 2>/dev/null)
    if [ "$c" -ge "$n" ]; then sleep 5; scancel $j; echo "$(date +%T) job $j: $c/$n checkpoints saved -> cancelled (codes on CPU)";
    elif [ -z "$st" ]; then echo "$(date +%T) job $j ended on its own with $c/$n checkpoints";
    else nxt+=("$jn"); fi
  done
  left=("${nxt[@]}"); sleep 20
done
