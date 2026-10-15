#!/bin/bash
# CRIM13 (Burgos-Artizzu et al. 2012), CaltechDATA record 4emt5-b0t10 (old id 1892 redirects there), CC-BY.
# Download -> size + md5 vs the record API (../flyvsfly/caltechdata_download.py, range-level resume) -> extract with
# CRC check (../flyvsfly/extract_zip.py) -> delete each zip once its extraction is complete.
# The archives mix views: every train/test zip holds <name>_t.seq (top), <name>_s.seq (side), <name>_t.txt (labels),
# <name>_t-track.mat, <name>_t-seek.mat. The record offers no top-only file, so the whole zip is downloaded and
# md5-verified, and the side-view members (_s.seq) are NOT extracted (they go when the zip is deleted).
# validation.zip is a re-split of 20 videos that are all in train/test already (same names + sizes, listed remotely
# 2026-10-15): only its non-video members are extracted; its .seq CRCs are compared to the main copies in inventory.
# Not downloaded: CRIM13_train_feats.zip, CRIM13_test_feats.zip (precomputed features).
# Data root (BeeGFS): /mnt/beegfs/locatgrp/shared/rcadei/benchmark/crim13 (= dataset/benchmark/crim13)
# Layout (as the record README asks): train/ (train1+train2), test/ (test1+test2), validation/{train,test}/,
#         info/, code/, res/, output/, raw/ (record listing, .ok markers, extraction reports).
# One CPU job per archive: FILES=CRIM13_train1.zip sbatch scripts/benchmark/crim13/download.sh
#SBATCH --job-name=crim_dl
#SBATCH --output=results/vision/benchmark_crim13/logs/%x_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
D=/mnt/beegfs/locatgrp/shared/rcadei/benchmark/crim13
S=scripts/benchmark/flyvsfly
PY=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python
mkdir -p $D/raw
echo "host $(hostname) job $SLURM_JOB_ID files $FILES start $(date)"
$PY -I $S/caltechdata_download.py 4emt5-b0t10 $D/raw $FILES
for f in $FILES; do
  [ -f $D/raw/$f ] || { echo "$f already extracted"; continue; }
  case $f in
    CRIM13_train?.zip) $PY -I $S/extract_zip.py $D/raw/$f $D/train $D/raw/extract_$f.json --skip '_s\.seq$' ;;
    CRIM13_test?.zip)  $PY -I $S/extract_zip.py $D/raw/$f $D/test $D/raw/extract_$f.json --skip '_s\.seq$' ;;
    validation.zip)    $PY -I $S/extract_zip.py $D/raw/$f $D $D/raw/extract_$f.json --skip '\.seq$' ;;
    CRIM13_info.zip)   $PY -I $S/extract_zip.py $D/raw/$f $D/info $D/raw/extract_$f.json ;;
    CRIM13_code.zip)   $PY -I $S/extract_zip.py $D/raw/$f $D/code $D/raw/extract_$f.json ;;
    CRIM13_res.zip)    $PY -I $S/extract_zip.py $D/raw/$f $D/res $D/raw/extract_$f.json ;;
    output.zip)        $PY -I $S/extract_zip.py $D/raw/$f $D/output $D/raw/extract_$f.json ;;
  esac
done
echo "ALLDONE $(date)"
