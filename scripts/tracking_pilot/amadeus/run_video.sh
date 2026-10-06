#!/bin/bash
# Run the full AMADEUS pilot for one video (steps 2-6; step 1 = prepare.sbatch must be done).
#
#   sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=1-00:00:00 \
#          --job-name=amd_<id> --output=<log> run_video.sh <video_id>
#
# Debug variant (short clip, tiny training) - set before calling:
#   SESSION_NAME=session_dbg ANALYSIS=<path to clip> EPOCHS=5 NUM_IMAGES=1000 run_video.sh <video_id>
set -euo pipefail
VID="$1"
HERE=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/tracking_pilot/amadeus
AMD=/nfs/scistore19/locatgrp/rcadei/tools/AMADEUS
PY=$AMD/.venv/bin/python
export PATH=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin:$PATH   # ffprobe
export YOLO_CONFIG_DIR=$AMD/.ultralytics MPLCONFIGDIR=$AMD/.mplconfig
cd "$HERE"
OD=$($PY -c "from pilot import out_dir; print(out_dir('$VID'))")
SESSION="$OD/${SESSION_NAME:-session}"
ANALYSIS="${ANALYSIS:-$OD/work/$VID.mp4}"
EPOCHS="${EPOCHS:-50}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$SESSION"
echo "[run_video] host=$(hostname) job=${SLURM_JOB_ID:-none} vid=$VID session=$SESSION analysis=$ANALYSIS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$SESSION/gpu.txt"

stamp() { date +%s; }
T0=$(stamp)
# 2. headless segmentation
if [ ! -f "$SESSION/segmentation/segmentation_gui_config.json" ]; then
  $PY segment.py "$VID" --analysis "$ANALYSIS" --session "$SESSION" --workers "$WORKERS"
fi
T1=$(stamp)
# 3. config (only if absent: initial_tracking writes auto-parameters back into it)
if [ ! -f "$SESSION/config.yaml" ]; then
  $PY make_config.py "$VID" --analysis "$ANALYSIS" --session "$SESSION" --variant idcorr \
      --workers "$WORKERS" --epochs "$EPOCHS" --num-images "$NUM_IMAGES"
fi
# 4. AMADEUS, all stages, overlap = very heavy (identity correction on)
$PY -u "$AMD/main/batch.py" "$SESSION/config.yaml"
T2=$(stamp)
# 5. second refinement pass without identity correction, from the same forward buffers
mkdir -p "$SESSION/noidcorr_pass"
$PY - "$SESSION" <<'EOF'
import sys, yaml
s = sys.argv[1]
cfg = yaml.safe_load(open(f"{s}/config.yaml"))
for k in list(cfg):
    if k.startswith("skip_"):
        cfg[k] = True
cfg["skip_refinement"] = False
cfg["EMBEDDING"]["ENABLE"] = False
yaml.safe_dump(cfg, open(f"{s}/noidcorr_pass/config.yaml", "w"), sort_keys=False)
EOF
$PY -u "$AMD/main/batch.py" "$SESSION/noidcorr_pass/config.yaml"
T3=$(stamp)
echo "{\"segment_sec\": $((T1-T0)), \"amadeus_main_sec\": $((T2-T1)), \"refinement_noidcorr_sec\": $((T3-T2))}" \
  > "$SESSION/wall_times.json"
# 6. normalise to tracks.parquet + meta.json, then quality numbers and PNG grids
$PY normalize.py "$VID" --session "$SESSION"
$PY quality.py "$VID" --session "$SESSION"
echo "[run_video] done $VID"
