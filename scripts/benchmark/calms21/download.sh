#!/bin/bash
# Download the selected CalMS21 files (CaltechDATA s0vdx-0k302, CC BY 4.0) with download.py (python requests, ranged,
# parallel), verify size + md5 against the record API,
# extract with python zipfile (CRC checked member by member), delete the zips that were fully extracted.
# Data root (BeeGFS, NFS is full): /mnt/beegfs/locatgrp/shared/rcadei/benchmark/calms21 (= dataset/benchmark/calms21).
# Selected: task1 MP4 videos (the only videos in the record), task1/2/3 pose + annotation JSONs, the readme, the npy
# converter. NOT downloaded: unlabeled_videos.zip (it holds keypoint / taskprog JSONs only, no video - listed remotely
# 2026-10-15), task1_videos_seq.zip (same task1 videos in the 74 GB seq format), task1_MARS_features.zip.
# CPU job: [FILES="a.zip b.zip"] sbatch scripts/benchmark/calms21/download.sh
#SBATCH --job-name=calms_dl
#SBATCH --output=results/vision/benchmark_calms21/logs/%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
D=/mnt/beegfs/locatgrp/shared/rcadei/benchmark/calms21

FILES=${FILES:-"readme.md calms21_convert_to_npy.py task1_classic_classification.zip task2_annotation_styles.zip task3_new_behaviors.zip task1_videos_mp4.zip"}
mkdir -p $D/raw $D/videos
echo "host $(hostname) start $(date)"
module load conda
conda activate crl
python -I scripts/benchmark/calms21/download.py $D $FILES
python -I scripts/benchmark/calms21/unzip.py $D $FILES
echo "ALLDONE $(date)"
