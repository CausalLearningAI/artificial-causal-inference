# AMADEUS tracking pilot (headless, SLURM)

Raw multi-animal tracks for 6 pilot videos (ants 3_1_1, 3_6_6, 3_21_2; mice rd25, rd32, rd18)
with [AMADEUS](https://github.com/jpmyrmecol/AMADEUS) v1.1.6, run without its GUI.

## Install (once)

AMADEUS pins numpy 1.26.4 / torch 2.14 / ultralytics 8.3.185, so it lives in its own folder and venv,
never in the project environment.

```bash
mkdir -p /nfs/scistore19/locatgrp/rcadei/tools && cd /nfs/scistore19/locatgrp/rcadei/tools
git clone --branch v1.1.6 --depth 1 https://github.com/jpmyrmecol/AMADEUS.git   # commit 55646c5c28a7...
cd AMADEUS
curl -LsSf https://astral.sh/uv/0.12.17/install.sh | env UV_INSTALL_DIR=$PWD/.uv INSTALLER_NO_MODIFY_PATH=1 sh
export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_INSTALL_DIR=$PWD/.uv-python UV_PROJECT_ENVIRONMENT=$PWD/.venv
.uv/uv sync --locked --python 3.10 --no-dev --extra cu126          # torch 2.14.0+cu126
.uv/uv pip install --python .venv/bin/python pyarrow==17.0.0        # only addition: parquet output
.venv/bin/python -c "from tools.ffmpeg_runtime import ensure_ffmpeg; print(ensure_ffmpeg())"  # pinned ffmpeg in .ffmpeg-hardware/
mkdir -p model && for w in yolo11n-obb.pt yolo11n.pt; do
  curl -sSLf -o model/$w https://github.com/ultralytics/assets/releases/download/v8.3.0/$w; done
```
The launcher `AMADEUS.sh` refuses to run without a display, so it is not used. No AMADEUS file is patched.

## Run

```bash
cd scripts/tracking_pilot/amadeus
L=../../../results/tracking_pilot/logs
# 1. frame-exact analysis copy (trim to window, mice scaled x0.5), CPU
for v in rd25 rd32 rd18 3_1_1 3_6_6 3_21_2; do sbatch --job-name=prep_$v prepare.sbatch $v; done
# 2-6. segmentation, AMADEUS (identity correction on), refinement without it, normalise, quality
for v in rd25 rd32 rd18 3_1_1 3_6_6 3_21_2; do
  sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=2-00:00:00 \
         --job-name=amd_$v --output=$L/run_${v}_%j.log run_video.sh $v; done
```
Debug run on a short clip (what validated the install, 2080 Ti, ~17 min):
```bash
python prepare_video.py 3_1_1 --max-frames 3600 --suffix _dbg120
SESSION_NAME=session_dbg ANALYSIS=<...>/work/3_1_1_dbg120.mp4 EPOCHS=5 NUM_IMAGES=1000 \
  sbatch --export=ALL --partition=debug_gpu --gres=gpu:1 --cpus-per-task=6 --mem=32G --time=01:00:00 run_video.sh 3_1_1
```

## Files

| file | what |
|---|---|
| `pilot.py` | the 6 videos, source windows, per-setup animal count and scale |
| `prepare_video.py` / `prepare.sbatch` | step 1: analysis copy `work/<id>.mp4` + `prep.json` (ffprobe facts, frame-alignment check) |
| `segment.py` | step 2: headless replacement for the segmentation GUI (writes the 3 files AMADEUS reads) |
| `make_config.py` | step 3: the `config.yaml` the Easy Tracking GUI would write ("Very heavy" overlap, backward movement = yes) |
| `run_video.sh` | steps 2-6 for one video, incl. the second refinement pass without identity correction |
| `normalize.py` | step 6a: `tracks.parquet`, `detections.parquet`, `meta.json`, `raw/` |
| `quality.py` | step 6b: `quality.json`, `quality_grid.png` |

Outputs: `results/tracking_pilot/{ants|mice}/<id>/amadeus/` (gitignored). The AMADEUS session
(synthetic images, trained detector, buffers) is in `session/`; `meta.json["conventions"]` documents
every column of `tracks.parquet`.

## Choices that replace GUI clicks
* Segmentation: ants = dark AND differs from background, ROI = Hough circle of the dish; mice = dark
  (incl. grey shave marks) AND differs from background, ROI = bedding bounding box. Background = median of
  100 frames with dark animal pixels masked out. Single-animal area bounds = [0.5, 1.5] x median blob area
  in frames where the blob count equals N. Same rule for all videos of a species (ROI found per video).
* Easy Tracking answers: N animals, overlap "Very heavy" (turns on contrastive identity correction),
  backward movement "yes". Detector trained per video (AMADEUS default), 10k synthetic images, 50 epochs.
* Mice tracked at 1032x1032 (x0.5) and native 30 fps; ants at native 824x824, 30 fps. Output coordinates
  are converted back to source pixels and source frame indices.
