"""
Foreground-only SAE training tokens (ECI, mice v1): 1 fps frames (every 5th frame, seeded
offset per video) of all 432 videos -> DINOv2-base on the whole frame at 448 (no crop) ->
keep only the foreground patches (src/eci/foreground.py FG_RULE, per-video backgrounds
from scripts/eci/fg_background.py). No behaviour annotation is used.

Output: dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1/
    shards/shard_XX/tokens.f16  raw float16 (n_tokens, 768)   (np.memmap, shape in shard.json)
                    row.i32     annotations.csv row of each token
                    pos.i16     patch position 0..1023 (row-major 32 x 32) of each token
                    frames.npz  rows, n_fg (foreground patches per frame), in frame order
                    shard.json, DONE
                    prev.f16    (with --motion-delta D > 0) raw float16 (n_tokens, 768): the token at
                                the SAME patch position D frames (5 fps) earlier, clipped to the
                                video's first frame (zero change there)
    (read with src.eci.foreground.FgTokenStore)
    --rule v3 uses FG_RULE_V3 (dilation only onto patches with dark pixels).

Usage:
    python scripts/eci/fg_extract_train.py --task 3 --n-tasks 16
    python scripts/eci/fg_extract_train.py --task 0 --n-tasks 16 --max-obs 1 --out-dir /some/test/dir
    python scripts/eci/fg_extract_train.py --task 3 --rule v3 --motion-delta 2 \
        --out-dir dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fgv3_fps1_d2
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.foreground import (RULES, FgBackgrounds, FrameDatasetFG, encode_batch,  # noqa: E402
                                load_encoder_fg, obs_rows)


def fps1_rows(ranges, stride=5, seed=0):
    """observation_id -> rows lo+offset::stride, offset ~ U{0..stride-1} per video (sorted ids)."""
    rng = np.random.default_rng(seed)
    return {o: np.arange(ranges[o][0] + int(rng.integers(0, stride)), ranges[o][1], stride) for o in sorted(ranges)}


class PairDataset(torch.utils.data.Dataset):
    """i -> (pix_t, grey_t, row_t, pix_prev): frame t and the frame D earlier (same video)."""

    def __init__(self, cur, prev):
        self.cur, self.prev = cur, prev

    def __len__(self):
        return len(self.cur)

    def __getitem__(self, i):
        pix, grey, r = self.cur[i]
        return pix, grey, r, self.prev[i][0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--bg-dir', default=str(REPO / 'dataset/mice/v1/eci/fg448/background'))
    p.add_argument('--out-dir', default=str(REPO / 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1'))
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--n-tasks', type=int, default=16)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-obs', type=int, default=None, help='testing')
    p.add_argument('--rule', default='fg448', choices=sorted(RULES))
    p.add_argument('--motion-delta', type=int, default=0, help='also store the token D frames earlier (prev.f16)')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=16)
    args = p.parse_args()

    ds, out = Path(args.dataset_dir), Path(args.out_dir)
    shard = out / 'shards' / f'shard_{args.task:02d}'
    if (shard / 'DONE').exists():
        print(f'[SKIP] {shard}')
        return
    tmp = Path(str(shard) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    ann = ds / 'mice/v1/annotations.csv'
    ranges = obs_rows(ann)
    sel = fps1_rows(ranges, args.stride, args.seed)
    ids = sorted(ranges)[args.task::args.n_tasks][:args.max_obs]
    rows = np.concatenate([sel[o] for o in ids])
    paths = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    print(f'task {args.task}/{args.n_tasks}: {len(ids)} videos, {len(rows)} frames', flush=True)

    device = torch.device('cuda')
    _, processor, model = load_encoder_fg(device=device)
    rule = RULES[args.rule]
    bgs = FgBackgrounds(args.bg_dir, ranges, rule, device)
    ds_cur = FrameDatasetFG([str(ds / pth) for pth in paths[rows]], processor, rows)
    D = args.motion_delta
    if D > 0:
        lo_of = np.concatenate([np.full(len(sel[o]), ranges[o][0]) for o in ids])
        prev_rows = np.maximum(rows - D, lo_of)
        dset = PairDataset(ds_cur, FrameDatasetFG([str(ds / pth) for pth in paths[prev_rows]], processor))
    else:
        dset = ds_cur
    loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=False, pin_memory=True, prefetch_factor=4)

    names = ('tokens.f16', 'row.i32', 'pos.i16') + (('prev.f16',) if D > 0 else ())
    files = [open(tmp / n, 'wb') for n in names]
    f_tok, f_row, f_pos = files[:3]
    n_fg_all, rows_seen, n_tok, t0 = [], [], 0, time.time()
    for b, batch in enumerate(loader):
        pix, grey, r = batch[:3]
        tok = encode_batch(model, pix, device)
        r = r.numpy()
        mask, _ = bgs.mask(tok, grey.to(device, non_blocking=True), r)
        fi, pi = torch.nonzero(mask, as_tuple=True)
        if D > 0:
            files[3].write(encode_batch(model, batch[3], device)[fi, pi].cpu().numpy().tobytes())
        f_tok.write(tok[fi, pi].cpu().numpy().tobytes())
        f_row.write(r[fi.cpu().numpy()].astype(np.int32).tobytes())
        f_pos.write(pi.cpu().numpy().astype(np.int16).tobytes())
        n_fg_all.append(mask.sum(1).cpu().numpy().astype(np.int16))
        rows_seen.append(r)
        n_tok += len(fi)
        if b % 50 == 0:
            el = time.time() - t0
            print(f'  batch {b:4d}  {sum(map(len, rows_seen)):6d}/{len(rows)} frames  {n_tok:,} tokens  '
                  f'{sum(map(len, rows_seen)) / el:.1f} f/s', flush=True)
    for f in files:
        f.close()
    rows_seen, n_fg = np.concatenate(rows_seen), np.concatenate(n_fg_all)
    assert np.array_equal(rows_seen, rows) and n_fg.sum() == n_tok
    np.savez(tmp / 'frames.npz', rows=rows, n_fg=n_fg)
    info = {'n_tokens': int(n_tok), 'n_frames': int(len(rows)), 'dim': int(tok.shape[-1]), 'observations': ids,
            'fg_frac_mean': float(n_fg.mean() / 1024), 'elapsed_s': round(time.time() - t0, 1), 'rule': rule,
            'rule_name': args.rule, 'motion_delta': D,
            'stride': args.stride, 'seed': args.seed}
    (tmp / 'shard.json').write_text(json.dumps(info, indent=1))
    if shard.exists():
        shutil.rmtree(shard)
    tmp.rename(shard)
    (shard / 'DONE').touch()
    print(f'Done: {len(rows)} frames, {n_tok:,} tokens ({n_tok / len(rows):.1f}/frame) in {info["elapsed_s"]}s -> {shard}')


if __name__ == '__main__':
    main()
