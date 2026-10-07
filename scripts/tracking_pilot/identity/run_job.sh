#!/bin/bash
# Identity layer + evaluation for one video, inside a SLURM job, on a /localhome staging copy.
#   mice (DINOv2 reads need a GPU):
#     sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=6:00:00 run_job.sh mice rd25
#   ants:
#     sbatch --partition=defaultp --cpus-per-task=8 --mem=32G --time=4:00:00 run_job.sh ants 3_1_1
# Runs: AMADEUS tracks (tag amadeus, full read pass + audit) and the stand-in tracks (tag standin,
# cached reads, metrics only), then copies identity/{amadeus,standin} outputs back to NFS once.
set -euo pipefail
DOM=$1; VID=$2
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
PY=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python
NFS=$REPO/results/tracking_pilot
STAGE=/localhome/$USER/${SLURM_JOB_ID:-manual$$}
trap 'rm -rf "$STAGE"' EXIT
mkdir -p $STAGE/videos $STAGE/results/$DOM/$VID/{amadeus,standin} $STAGE/results/$DOM/$VID/identity/standin
SRC=$(cd $REPO/scripts/tracking_pilot/identity && $PY -c "from common import video_path; print(video_path('$DOM','$VID'))")
cp "$SRC" $STAGE/videos/
cp $NFS/$DOM/$VID/amadeus/tracks.parquet $STAGE/results/$DOM/$VID/amadeus/
cp $NFS/$DOM/$VID/standin/tracks.parquet $STAGE/results/$DOM/$VID/standin/
cp -r $NFS/$DOM/scorer $STAGE/results/$DOM/
cp $NFS/$DOM/$VID/identity/background_$DOM.npy $STAGE/results/$DOM/$VID/identity/
for f in feats.parquet dino.npy identity_tracks_standin_blob.parquet; do
  [ -f $NFS/$DOM/$VID/identity/standin/$f ] && cp $NFS/$DOM/$VID/identity/standin/$f $STAGE/results/$DOM/$VID/identity/standin/
done
mv $STAGE/results/$DOM/$VID/identity/standin/identity_tracks_standin_blob.parquet $STAGE/old_standin_blob.parquet
mkdir -p $STAGE/results/$DOM/$VID/identity/amadeus
for f in feats.parquet dino.npy; do   # cached AMADEUS mark reads from an earlier job (skips the video pass)
  [ -f $NFS/$DOM/$VID/identity/amadeus/$f ] && cp $NFS/$DOM/$VID/identity/amadeus/$f $STAGE/results/$DOM/$VID/identity/amadeus/
done
export TP_RESULTS=$STAGE/results TP_VIDEO_DIR=$STAGE/videos HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd $REPO/scripts/tracking_pilot/identity
echo "== $DOM/$VID on $(hostname) staging $STAGE"; nvidia-smi -L 2>/dev/null || true
$PY run.py --domain $DOM --video $VID --tracks $STAGE/results/$DOM/$VID/amadeus/tracks.parquet --tag amadeus
if [ "${SKIP_STANDIN:-0}" != 1 ]; then
$PY run.py --domain $DOM --video $VID --tracks $STAGE/results/$DOM/$VID/standin/tracks.parquet --tag standin --no_audit
$PY - <<PYEOF
import pandas as pd
a = pd.read_parquet('$STAGE/old_standin_blob.parquet'); b = pd.read_parquet('$STAGE/results/$DOM/$VID/identity/standin/identity_tracks_standin_blob.parquet')
cols = ['frame_src', 'identity', 'track_id', 'state']
same = a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True))
print('standin_blob identity output identical to the audited version:', same,
      '' if same else f"(state agreement {(a.state.values == b.state.values).mean():.4f})")
PYEOF
fi
mkdir -p $NFS/$DOM/$VID/identity/amadeus
rm -rf $NFS/$DOM/$VID/identity/amadeus/audit   # stale renders of a previous run
cp -r $STAGE/results/$DOM/$VID/identity/amadeus/. $NFS/$DOM/$VID/identity/amadeus/
[ "${SKIP_STANDIN:-0}" != 1 ] && for f in metrics.json reads.parquet identity_tracks_standin_blob.parquet identity_tracks_standin_anttracker.parquet tracklets_standin_blob.parquet tracklets_standin_anttracker.parquet; do
  [ -f $STAGE/results/$DOM/$VID/identity/standin/$f ] && cp $STAGE/results/$DOM/$VID/identity/standin/$f $NFS/$DOM/$VID/identity/standin/
done
echo "== done $DOM/$VID"
