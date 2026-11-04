#!/bin/bash
# v2.2 FULL WINDOW array wrapper: sbatch --array=0-(N-1) v22_full_array.sh <step> <N> [extra args]; task = SLURM_ARRAY_TASK_ID
STEP=$1; N=$2; shift 2
exec bash /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2/v22_run.sh v22_full.py $STEP --task $SLURM_ARRAY_TASK_ID --ntask $N "$@"
