#!/bin/bash
# Fly-vs-Fly (Eyjolfsdottir et al. 2014), CaltechDATA record zrznw-w7386 (old id 1893 redirects there), CC0.
# Download -> size + md5 vs the record API (caltechdata_download.py, range-level resume) -> extract zips with CRC check
# (extract_zip.py) -> delete each zip once its extraction is complete.
# Data root (BeeGFS; NFS is full): /mnt/beegfs/locatgrp/shared/rcadei/benchmark/flyvsfly (= dataset/benchmark/flyvsfly)
# Layout: Aggression/, Courtship/{hyper,wild}/, BoyMeetsBoy/ (the zip's "Boy_meets_boy light/", renamed: no space),
#         BoyMeetsBoy/movieN/movieN.avi (the 6 AVIs of the record), raw/ (record listing, .ok markers, reports).
# One CPU job per file:  for f in movie{1..6}.avi; do FILES=$f sbatch scripts/benchmark/flyvsfly/download.sh; done
#                        FILES="Aggression.zip Courtship.zip BoyMeetsBoy_wo_movies.zip visualizer.m" sbatch ...
#SBATCH --job-name=fvf_dl
#SBATCH --output=results/vision/benchmark_flyvsfly/logs/%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
D=/mnt/beegfs/locatgrp/shared/rcadei/benchmark/flyvsfly
S=scripts/benchmark/flyvsfly
PY=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python
mkdir -p $D/raw
echo "host $(hostname) job $SLURM_JOB_ID files $FILES start $(date)"
$PY -I $S/caltechdata_download.py zrznw-w7386 $D/raw $FILES
JUNK='(^__MACOSX/|\.DS_Store$)'
for f in $FILES; do
  case $f in
    movie?.avi) n=${f%.avi}; mkdir -p $D/BoyMeetsBoy/$n
                if [ -f $D/raw/$f ]; then mv $D/raw/$f $D/BoyMeetsBoy/$n/$f; fi
                ls -l $D/BoyMeetsBoy/$n/$f ;;
    Aggression.zip) [ -f $D/raw/$f ] && $PY -I $S/extract_zip.py $D/raw/$f $D/Aggression $D/raw/extract_$f.json --skip "$JUNK" --strip 1 ;;
    Courtship.zip) [ -f $D/raw/$f ] && $PY -I $S/extract_zip.py $D/raw/$f $D/Courtship $D/raw/extract_$f.json --skip "$JUNK" --strip 1 ;;
    BoyMeetsBoy_wo_movies.zip) [ -f $D/raw/$f ] && $PY -I $S/extract_zip.py $D/raw/$f $D/BoyMeetsBoy $D/raw/extract_$f.json --skip "$JUNK" --strip 1 ;;
    visualizer.m) cp $D/raw/$f $D/$f ;;
  esac
done
echo "ALLDONE $(date)"
