"""
Foreground-only SAE training tokens (ECI, mice v1 by default): 1 fps frames (every 5th frame, seeded
offset per video) of all 432 videos -> DINOv2-base on the whole frame at 448 (no crop) ->
keep only the foreground patches (src/eci/foreground.py FG_RULE, per-video backgrounds
from scripts/eci/fg_background.py). No behaviour annotation is used.
--domain (src/eci/domain.py) picks annotations.csv and the default dirs (ants: dataset/ants/eci/...);
--rule defaults to the domain rule ('all' = every patch, no backgrounds needed). --patch-frac f < 1
keeps a seeded random fraction f of each frame's foreground patches (whole-frame stores stay small);
frames.npz n_fg then counts the KEPT patches and shard.json records patch_frac.

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
    --encoder dinov3_base: the mask is still computed from DINOv2 (448), so the frames and the kept patches are the
    same as for DINOv2; the stored tokens are DINOv3 ViT-B/16 patch tokens of the whole 512 px frame (32 x 32 grid,
    CLS + registers dropped). Needs --out-dir; shard.json records 'encoder' (absent = dinov2_base). No --motion-delta.
    --align odor (mice): frames rotated so the odor corner is at the top right (src/eci/foreground.py align_rot90);
    backgrounds default to fg448al/background (must be aligned too), output default
    train_tokens/dinov2_base_l-1_fg448al_fps1; shard.json records 'align' (absent = none). Same frames as fg448.

Usage:
    python scripts/eci/fg_extract_train.py --task 3 --n-tasks 16
    python scripts/eci/fg_extract_train.py --task 0 --n-tasks 16 --max-obs 1 --out-dir /some/test/dir
    python scripts/eci/fg_extract_train.py --task 3 --rule v3 --motion-delta 2 \
        --out-dir dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fgv3_fps1_d2
    python scripts/eci/fg_extract_train.py --domain ants --rule all --patch-frac 0.25 --task 0 --n-tasks 16 \
        --out-dir dataset/ants/eci/train_tokens/dinov2_base_l-1_all448_fps1
    python scripts/eci/fg_extract_train.py --align odor --task 3
    python scripts/eci/fg_extract_train.py --encoder dinov3_base --task 3 \
        --out-dir dataset/mice/v1/eci/train_tokens/dinov3_base_l-1_fg512_fps1
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
from src.eci.domain import DOMAINS, get_domain  # noqa: E402
from src.eci.foreground import (ALIGNS, ENCODERS, MASK_ENCODER, RULES, FgBackgrounds, FgEncoder,  # noqa: E402
                                FrameDatasetFG, align_rot90, align_tag, encode_batch, obs_rows)


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
    p.add_argument('--domain', default='mice', choices=DOMAINS)
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--bg-dir', default=None, help='default <dataset dir>/<domain eci dir>/fg448[al]/background')
    p.add_argument('--out-dir', default=None,
                   help='default <dataset dir>/<domain eci dir>/train_tokens/dinov2_base_l-1_fg448[al]_fps1')
    p.add_argument('--align', default='none', choices=ALIGNS, help='odor: rotate frames, odor corner top right')
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--n-tasks', type=int, default=16)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-obs', type=int, default=None, help='testing')
    p.add_argument('--rule', default=None, choices=sorted(RULES), help='default: the domain rule (mice fg448)')
    p.add_argument('--patch-frac', type=float, default=1.0, help='keep this random fraction of the foreground patches')
    p.add_argument('--motion-delta', type=int, default=0, help='also store the token D frames earlier (prev.f16)')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--encoder', default=MASK_ENCODER, choices=ENCODERS,
                   help='encoder of the stored tokens (the mask always uses DINOv2 at 448)')
    args = p.parse_args()
    if args.encoder != MASK_ENCODER and (args.out_dir is None or args.motion_delta > 0):
        raise SystemExit(f'--encoder {args.encoder} needs --out-dir and does not support --motion-delta')

    dom = get_domain(args.domain)
    args.rule = args.rule or dom.fg_rule
    ds = Path(args.dataset_dir)
    tag = align_tag(args.align)
    args.bg_dir = args.bg_dir or str(ds / dom.eci_rel / tag / 'background')
    out = Path(args.out_dir) if args.out_dir else ds / dom.eci_rel / f'train_tokens/dinov2_base_l-1_{tag}_fps1'
    shard = out / 'shards' / f'shard_{args.task:02d}'
    if (shard / 'DONE').exists():
        print(f'[SKIP] {shard}')
        return
    tmp = Path(str(shard) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    ann = ds / dom.ann_rel
    ranges = obs_rows(ann)
    rot_all = align_rot90(args.align, ann)
    sel = fps1_rows(ranges, args.stride, args.seed)
    ids = sorted(ranges)[args.task::args.n_tasks][:args.max_obs]
    rows = np.concatenate([sel[o] for o in ids])
    paths = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    print(f'task {args.task}/{args.n_tasks}: {len(ids)} videos, {len(rows)} frames', flush=True)

    device = torch.device('cuda')
    enc = FgEncoder(args.encoder, device)
    processor, model = enc.processor, enc.model
    rule = RULES[args.rule]
    bgs = FgBackgrounds(args.bg_dir, ranges, rule, device, align=args.align)
    ds_cur = enc.dataset([str(ds / pth) for pth in paths[rows]], rows, None if rot_all is None else rot_all[rows])
    D = args.motion_delta
    if D > 0:
        lo_of = np.concatenate([np.full(len(sel[o]), ranges[o][0]) for o in ids])
        prev_rows = np.maximum(rows - D, lo_of)
        dset = PairDataset(ds_cur, FrameDatasetFG([str(ds / pth) for pth in paths[prev_rows]], processor,
                                                  rot=None if rot_all is None else rot_all[prev_rows]))
    else:
        dset = ds_cur
    loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=False, pin_memory=True, prefetch_factor=4)

    names = ('tokens.f16', 'row.i32', 'pos.i16') + (('prev.f16',) if D > 0 else ())
    files = [open(tmp / n, 'wb') for n in names]
    f_tok, f_row, f_pos = files[:3]
    keep_rng = np.random.default_rng([args.seed, args.task])
    n_fg_all, rows_seen, n_tok, t0 = [], [], 0, time.time()
    for b, batch in enumerate(loader):
        pix, grey, r = batch[:3]
        tok = encode_batch(model, pix, device)
        r = r.numpy()
        mask, _ = bgs.mask(tok, grey.to(device, non_blocking=True), r)
        if args.patch_frac < 1:  # seeded per shard; the same draw for a rerun of the shard
            mask &= torch.from_numpy(keep_rng.random(tuple(mask.shape)) < args.patch_frac).to(mask.device)
        if enc.model2 is not None:  # stored tokens from the second encoder, same frames and patches
            tok = enc.sae_tokens(tok, batch[3])
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
            'rule_name': args.rule, 'motion_delta': D, 'patch_frac': args.patch_frac, 'domain': args.domain,
            'stride': args.stride, 'seed': args.seed}
    if args.encoder != MASK_ENCODER:
        info['encoder'] = args.encoder
    if args.align != 'none':
        info['align'] = args.align
    (tmp / 'shard.json').write_text(json.dumps(info, indent=1))
    if shard.exists():
        shutil.rmtree(shard)
    tmp.rename(shard)
    (shard / 'DONE').touch()
    print(f'Done: {len(rows)} frames, {n_tok:,} tokens ({n_tok / len(rows):.1f}/frame) in {info["elapsed_s"]}s -> {shard}')


if __name__ == '__main__':
    main()
