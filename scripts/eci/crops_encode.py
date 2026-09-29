"""
Crop prototype step 1: dark-blob instances -> single / pair / merged crops -> DINOv2-base (224)
CLS token and mean patch token per crop (src/eci/crops.py; params from crops_select.py).
No behaviour label is read here.

Splits:
    train   the 20 training videos of config.json, every 5th row (1 fps)
    eval    every 5 fps frame of the annotated videos of the 8 held-out pools (the same 108k rows as
            dataset/mice/v1/eci/fg448/eval_codes)
Output: dataset/mice/v1/eci/crops/<split>/task_XX.npz
    rows (F,), geom (F, 5) [n_blobs, n_single, n_merged, n_fragment, min_dist]
    crop_row (n,), crop_spec (n, 5) [type, cy, cx, angle, side], crop_blobs (n, 2), cls (n, 768) fp16, mean (n, 768) fp16

Usage: python scripts/eci/crops_encode.py --split eval --task 0 --n-tasks 6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from src.eci.crops import IMNET_MEAN, IMNET_STD, CropFrameDataset, collate, sample_crops  # noqa: E402
from src.eci.foreground import obs_rows  # noqa: E402

CROPS = REPO / 'dataset/mice/v1/eci/crops'


def split_rows(split, cfg, ranges):
    if split == 'train':
        return np.concatenate([np.arange(*ranges[o])[::5] for o in cfg['train_videos']])
    from fg_eval_encode import eval_videos
    vids = eval_videos(REPO / 'data', cfg['val_pools'])
    return np.concatenate([np.arange(*ranges[o]) for o in vids])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=('train', 'eval'), required=True)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--n-tasks', type=int, default=1)
    p.add_argument('--batch-frames', type=int, default=48)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--limit', type=int, default=None, help='testing: first N frames of the task')
    args = p.parse_args()
    cfg = json.loads((CROPS / 'config.json').read_text())
    out = CROPS / args.split / f'task_{args.task:02d}.npz'
    if out.exists():
        print(f'[SKIP] {out}'); return
    out.parent.mkdir(parents=True, exist_ok=True)
    ann = REPO / 'dataset/mice/v1/annotations.csv'
    ranges = obs_rows(ann)
    rows = split_rows(args.split, cfg, ranges)
    rows = np.array_split(rows, args.n_tasks)[args.task]
    if args.limit:
        rows = rows[:args.limit]
    meta = pd.read_csv(ann, usecols=['observation_id', 'frame_path'])
    ds = CropFrameDataset(rows, meta.frame_path.values, meta.observation_id.values,
                          REPO / 'dataset/mice/v1/eci/fg448/background', REPO / 'dataset', cfg['params'])
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_frames, num_workers=args.workers, collate_fn=collate,
                                     prefetch_factor=4, persistent_workers=False)
    from transformers import AutoModel
    dev = torch.device('cuda')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(dev).eval().requires_grad_(False)
    mean_, std_ = IMNET_MEAN.to(dev), IMNET_STD.to(dev)
    R, G, CR, CS, CB, CLS, MEAN = [], [], [], [], [], [], []
    t0, nf = time.time(), 0
    print(f'{args.split} task {args.task}: {len(rows)} frames', flush=True)
    for rgb, crops, bidx, fi, geom, rr in dl:
        rgb = rgb.to(dev, non_blocking=True)
        for a in range(0, len(crops), 256):
            x = sample_crops(rgb, crops[a:a + 256], fi[a:a + 256])
            x = (x - mean_) / std_
            with torch.inference_mode():
                hs = model(pixel_values=x).last_hidden_state.float()
            CLS.append(hs[:, 0].half().cpu().numpy()); MEAN.append(hs[:, 1:].mean(1).half().cpu().numpy())
        R.append(rr); G.append(geom.numpy()); CR.append(rr[fi.numpy()]); CS.append(crops.numpy()); CB.append(bidx.numpy())
        nf += len(rr)
        if len(R) % 50 == 0:
            print(f'  {nf}/{len(rows)} frames, {sum(len(c) for c in CR)} crops, {time.time() - t0:.0f}s', flush=True)
    arr = dict(rows=np.concatenate(R), geom=np.concatenate(G), crop_row=np.concatenate(CR),
               crop_spec=np.concatenate(CS), crop_blobs=np.concatenate(CB),
               cls=np.concatenate(CLS) if CLS else np.zeros((0, 768), np.float16),
               mean=np.concatenate(MEAN) if MEAN else np.zeros((0, 768), np.float16))
    assert len(arr['cls']) == len(arr['crop_row']) and np.isfinite(arr['cls'].astype(np.float32)).all()
    tmp = str(out).replace('.npz', '.tmp.npz')
    np.savez(tmp, **arr); Path(tmp).rename(out)
    print(f'Done: {nf} frames, {len(arr["crop_row"])} crops (types {np.bincount(arr["crop_spec"][:, 0].astype(int), minlength=3)}) '
          f'in {time.time() - t0:.0f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()
