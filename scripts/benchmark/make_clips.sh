#!/bin/bash
#SBATCH --job-name=mk_clips
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/vision/benchmark_examples/clips/slurm_%j.log
# Usage: sbatch scripts/benchmark/make_clips.sh cands   (contact sheets of candidate frames)
#        sbatch scripts/benchmark/make_clips.sh clips   (final 3 s clips, starts from STARTS below)
# ffmpeg from the crl env (has libx264); WebP is built with img2webp (libwebp) from the same env,
# because the ffmpeg611 env has no libwebp and no libx264.
set -euo pipefail
MODE=${1:-clips}
R=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
B=$R/dataset/benchmark
BIN=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin
OUT=$R/results/vision/benchmark_examples/clips
W=/localhome/$USER/${SLURM_JOB_ID:-manual}
mkdir -p $W $OUT; trap 'rm -rf $W' EXIT
declare -A SRC START
SRC[mabe_mouse]=$B/mabe22/videos/mouse_submission/089DVOZHBVT7B4CMXSZ9.avi
SRC[mabe_beetle]=$B/mabe22/videos/beetle_submission/0006222c3deef773b787.avi
SRC[calms21]=$B/calms21/videos/task1_videos_mp4/train/mouse001_task1_annotator1.mp4
SRC[lab_ants]=$R/data/ants/v3/observations/source/3_10_1.mkv
SRC[lab_mice]=$R/data/mice/source/2024-07-22_12-35-04_BHVScreen_rd11_1_SocialOdor_Test.mp4
SRC[lab_frogs]=/nfs/scistore23/sweengrp/ftoma/Behavior_MNV1/MNV1_revision_dataset/Juv/WT/157_2023-01-17_14-46-00_WT_stJuv_1_1.mp4
# candidate times (s) for contact sheets
declare -A CAND
CAND[mabe_mouse]="2 10 18 26 34 42 50 56"
CAND[mabe_beetle]="2 6 10 14 18 22 25 27"
CAND[calms21]="60 150 240 330 420 510 600 690"
CAND[lab_ants]="60 120 180 240 300 360 420 480"
CAND[lab_mice]="60 150 240 330 420 510 600 700"
CAND[lab_frogs]="120 600 1200 1800 2400 3000 3300 3500"
# chosen start times (s), filled in after looking at the sheets
START[mabe_mouse]=${S_mabe_mouse:-15}
START[mabe_beetle]=${S_mabe_beetle:-18}
START[calms21]=${S_calms21:-60}
START[lab_ants]=${S_lab_ants:-240}
START[lab_mice]=${S_lab_mice:-330}
START[lab_frogs]=${S_lab_frogs:-120}
NAMES="mabe_mouse mabe_beetle calms21 lab_ants lab_mice lab_frogs"
if [ $MODE = cands ]; then
  for n in $NAMES; do
    for t in ${CAND[$n]}; do
      $BIN/ffmpeg -v error -y -ss $t -i "${SRC[$n]}" -frames:v 1 -vf "scale=240:240:force_original_aspect_ratio=decrease,pad=240:240:(ow-iw)/2:(oh-ih)/2" $W/${n}_$(printf %05d $t).png
    done
    $BIN/ffmpeg -v error -y -pattern_type glob -framerate 1 -i "$W/${n}_*.png" -vf "tile=4x2" -frames:v 1 -q:v 3 $OUT/cand_$n.jpg
  done
  exit 0
fi
echo "[" > $W/clips.json; first=1
for n in $NAMES; do
  s=${START[$n]}; src=${SRC[$n]}
  mkdir -p $W/$n
  # 3 s, 15 fps, long side 360, original colour (no grey conversion)
  $BIN/ffmpeg -v error -y -ss $s -i "$src" -t 3 -vf "fps=15,scale=360:360:force_original_aspect_ratio=decrease:flags=area" -frames:v 45 $W/$n/f%02d.png
  [ $(ls $W/$n | wc -l) -eq 45 ] || { echo "FAIL $n frame count"; exit 1; }
  $BIN/img2webp -loop 0 -lossy -q 65 -m 6 -d 67 $W/$n/f*.png -o $W/$n.webp >/dev/null
  $BIN/ffmpeg -v error -y -framerate 15 -i $W/$n/f%02d.png -c:v libx264 -pix_fmt yuv420p -crf 30 -preset slow -movflags +faststart -an $W/$n.mp4
  $BIN/ffmpeg -v error -y -i $W/$n/f01.png -q:v 3 $W/$n.jpg
  for e in webp mp4 jpg; do cp $W/$n.$e $OUT/; done
  dims=$($BIN/ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$src")
  od=$($BIN/ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $W/$n.mp4)
  [ $first = 1 ] || echo "," >> $W/clips.json; first=0
  cat >> $W/clips.json <<J
{"name":"$n","source":"$src","start_s":$s,"native_w_h_fps":"$dims","out_w_h":"$od","out_fps":15,"frames":45,
 "bytes":{"webp":$(stat -c %s $W/$n.webp),"mp4":$(stat -c %s $W/$n.mp4),"jpg":$(stat -c %s $W/$n.jpg)}}
J
done
echo "]" >> $W/clips.json
cp $W/clips.json $OUT/clips.json
