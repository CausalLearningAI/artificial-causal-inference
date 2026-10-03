"""
Frames of ONE frogs v1 video (SLURM array task = row of data/frogs/v1/experiment.csv), with the stage-0 / stage-1
functions of mice and ants: src/data/standardize.py process_video (5 fps, 512 x 512, libopenh264 2M; mirrored
left-right first when experiment.csv 'hflip' = 1) -> data/frogs/v1/observations/full/<id>.mp4, then
src/dataset/get_frames.py extract_frames -> dataset/frogs/v1/frames/full/<id>/frame_%06d.jpg (JPEG q 2).
Same commands as `python src/data/standardize.py experiment=frogs/v1` + `python src/dataset/get_frames.py
experiment=frogs/v1`, one video per task so the 35 videos run in parallel. Finished outputs are skipped.

Usage: python scripts/eci/frogs_frames.py --task 3
"""
import argparse
import csv
import sys
import time
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data.standardize import process_video  # noqa: E402
from src.dataset.get_frames import extract_frames  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, required=True)
    args = p.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(REPO / 'configs')):
        cfg = compose(config_name='config', overrides=['experiment=frogs/v1'])
    rows = [r for r in csv.DictReader(open(REPO / 'data/frogs/v1/experiment.csv')) if r['valid'] == '1']
    r = rows[args.task]
    src = REPO / 'data/frogs/v1/observations/source' / r['observation_file']
    vid = REPO / 'data/frogs/v1/observations/full' / r['observation_file']
    frames = REPO / 'dataset/frogs/v1/frames/full' / vid.stem
    vid.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not vid.exists():
        tmp = vid.with_name(vid.stem + '.tmp.mp4')
        tmp.unlink(missing_ok=True)
        if not process_video(src, tmp, int(r['start_frame']), int(r['end_frame']), cfg, hflip=r['hflip'] == '1'):
            raise SystemExit(f'{src}: standardize failed')
        tmp.rename(vid)
    print(f'{r["observation_id"]} (hflip {r["hflip"]}): video {time.time() - t0:.0f}s', flush=True)
    done = frames / 'DONE'
    if not done.exists():
        if not extract_frames(vid, frames, overwrite=True, fps=cfg.data.target_fps, frame_format=cfg.data.frame_format):
            raise SystemExit(f'{vid}: frame extraction failed')
        done.touch()
    print(f'{r["observation_id"]}: {len(list(frames.glob("frame_*.jpg")))} frames, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
