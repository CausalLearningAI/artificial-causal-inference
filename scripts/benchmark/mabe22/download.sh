#!/bin/bash
# Download the selected MABe22 files (CaltechDATA rdsa8-rde65), verify md5 vs the record, unzip videos.
# CPU job: sbatch scripts/benchmark/mabe22/download.sh
#SBATCH --job-name=mabe_dl
#SBATCH --output=results/vision/benchmark_mabe22/logs/%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
D=dataset/benchmark/mabe22
API=https://data.caltech.edu/api/records/rdsa8-rde65/files
FILES="mouse_userTrain_videos_resized_224.zip mouse_submission_videos_resized_224.zip mouse_user_train.npy mouse_submission_keypoints.npy mouse_submission_labels.npy mouse_triplets_test_labels.npy mouse_frame_number_map.npy mouse_sample_submission.npy beetle_userTrain_videos_resized_224.zip beetle_submission_videos_resized_224.zip beetle_user_train.npy beetle_submission_keypoints.npy beetle_submission_labels.npy beetle_frame_number_map.npy beetle_sample_submission.npy fly_groups_test_labels.npy fly_frame_number_map.npy"
curl -sf $API -o $D/raw/record_files.json
for f in $FILES; do
  read size md5 < <(python3 -I -c "
import json,sys
for e in json.load(open('$D/raw/record_files.json'))['entries']:
    if e['key']=='$f': print(e['size'], e['checksum'].split(':')[1])")
  p=$D/raw/$f
  if [ ! -f $p ] || [ "$(stat -c %s $p)" != "$size" ]; then
    curl -sfL --retry 5 -C - $API/$f/content -o $p
  fi
  got=$(md5sum $p | cut -d' ' -f1)
  if [ "$got" = "$md5" ] && [ "$(stat -c %s $p)" = "$size" ]; then echo "VERIFIED $f $size $md5"; else echo "FAILED $f"; exit 1; fi
done
exit 0  # unzip moved to unzip.py
for z in mouse_userTrain mouse_submission beetle_userTrain beetle_submission; do
  mkdir -p $D/videos/$z
  unzip -q -o $D/raw/${z}_videos_resized_224.zip -d $D/videos/$z
  unzip -l $D/raw/${z}_videos_resized_224.zip | tail -1
  echo "unzipped $z"
done
echo ALLDONE
