#!/bin/bash
# Quick look before the inventory: ffprobe one video per context and dump one frame each to PNG (small, for tuning
# the segmentation thresholds). CPU job: sbatch scripts/benchmark/flyvsfly/peek.sh <out dir>
#SBATCH --job-name=fvf_peek
#SBATCH --output=results/vision/benchmark_flyvsfly/logs/%x_%j.out
#SBATCH --time=00:20:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
set -euo pipefail
D=/mnt/beegfs/locatgrp/shared/rcadei/benchmark/flyvsfly
B=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin
OUT=$1; mkdir -p $OUT
for v in Aggression/movie1/movie1.mp4 Courtship/wild/movie1/movie1.mp4 Courtship/hyper/movie1/movie1.mp4 BoyMeetsBoy/movie1/movie1.avi; do
  echo "== $v"; $B/ffprobe -v error -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,bit_rate -show_entries format=duration,size -of compact "$D/$v"
  $B/ffmpeg -v error -y -ss 60 -i "$D/$v" -frames:v 1 "$OUT/$(echo $v | tr / _).png"
done
