#!/bin/bash
#SBATCH --job-name=crim_clip
#SBATCH --partition=defaultp
#SBATCH --exclude=eta296,eta300
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=results/vision/benchmark_examples/clips/slurm_crim13_%j.log
# CRIM13 preview clip, same encoding as scripts/benchmark/make_clips.sh (3 s, 15 fps, 45 frames, long side 360,
# img2webp -lossy -q 65 -m 6 -d 67). Source frames are every 5/3 of a 25 fps frame (rounded). Frames are grey (the
# CRIM13 .seq files are 8-bit grey only). Segment: a labelled 'sniff' bout (frames 1654-1991 1-based) of the BC-intruder
# video 012609_A29_Block12_BCma1 (train), clip = 0-based frames 1720..1794.
set -euo pipefail
R=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
SEQ=${SEQ:-$R/dataset/benchmark/crim13/train/012609_A29_Block12_BCma1_t.seq}
START=${START:-1720}
BIN=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin
OUT=$R/results/vision/benchmark_examples/clips
W=/localhome/$USER/${SLURM_JOB_ID:-manual}
mkdir -p $W/f; trap 'rm -rf $W' EXIT
$BIN/python $R/scripts/benchmark/crim13/make_clip.py clip $SEQ $W/f $START
[ $(ls $W/f | wc -l) -eq 45 ]
$BIN/img2webp -loop 0 -lossy -q 65 -m 6 -d 67 $W/f/f*.png -o $W/crim13.webp >/dev/null
cp $W/crim13.webp $OUT/
