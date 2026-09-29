"""
Encode mice v1 frames with DINOv2 (whole frame, 448, no crop) + foreground mask + one or
more foreground SAEs (ECI, "mouse patches" pipeline, src/eci/foreground.py).

Per frame and SAE, pooled over the FOREGROUND patches only:
    codes_max   max over foreground patches    (N, n_latents) float16 (0 if no foreground)
    codes_mean  mean over foreground patches   (N, n_latents) float16 (0 if no foreground)
    n_fg        number of foreground patches   (N,) int16
Sharded by row ranges of annotations.csv, same resumable layout as src/eci/encode.py
(shards/shard_XX.tmp -> shards/shard_XX + DONE; merge_fg_shards; verify_fg_codes).

Functions:
    fg_sae_pool        (B, 1024, d) tokens + mask -> max / mean pooled codes over the mask
    encode_rows        DINOv2 + mask + SAEs on arbitrary rows -> dict of arrays (in memory)
    encode_fg_shard    rows [lo, hi) -> shard directory (memmaps)
    merge_fg_shards    shards -> final arrays + DONE
    verify_fg_codes    row count, non-finite rows, alignment with codes recomputed from scratch
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eci.foreground import FG_RULE, FgBackgrounds, FrameDatasetFG, encode_batch, load_encoder_fg, obs_rows
from src.eci.sae import load_sae

FG_OUTPUTS = ('codes_max', 'codes_mean', 'n_fg')


@torch.no_grad()
def fg_sae_pool(sae, norm, tokens, mask):
    """tokens (B, P, d) fp16, mask (B, P) bool -> (max (B, m), mean (B, m)) float32 over the
    masked patches (zeros for frames without foreground). Only masked tokens are encoded."""
    B, P, _ = tokens.shape
    fi, pi = torch.nonzero(mask, as_tuple=True)
    z = sae.encode(norm(tokens[fi, pi]), mode='threshold')  # (n, m), >= 0
    m = z.shape[1]
    mx = torch.zeros(B, m, device=z.device).index_reduce_(0, fi, z, 'amax', include_self=True)
    sm = torch.zeros(B, m, device=z.device).index_add_(0, fi, z)
    n = mask.sum(1, keepdim=True).float()
    return mx, sm / n.clamp_min(1)


class _Runner:
    def __init__(self, sae_paths, bg_dir, ann_path, device='cuda', rule=FG_RULE):
        self.device = torch.device(device)
        _, self.processor, self.model = load_encoder_fg(device=self.device)
        self.saes = [load_sae(p, self.device)[:2] for p in sae_paths]
        self.bgs = FgBackgrounds(bg_dir, obs_rows(ann_path), rule, self.device)

    def loader(self, paths, rows, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            FrameDatasetFG(paths, self.processor, rows), batch_size=batch_size, num_workers=num_workers,
            shuffle=False, pin_memory=self.device.type == 'cuda', prefetch_factor=4 if num_workers > 0 else None)

    def batch(self, pix, grey, rows):
        tok = encode_batch(self.model, pix, self.device)
        mask, _ = self.bgs.mask(tok, grey.to(self.device, non_blocking=True), rows)
        return mask, [fg_sae_pool(s, n, tok, mask) for s, n in self.saes]


def encode_rows(rows, frame_paths, sae_paths, bg_dir, ann_path, dataset_dir='dataset', batch_size=128,
                num_workers=16, device='cuda'):
    """-> list (one per SAE) of dicts codes_max / codes_mean (N, m) float16, plus n_fg (N,)."""
    run = _Runner(sae_paths, bg_dir, ann_path, device)
    rows = np.asarray(rows)
    paths = [str(Path(dataset_dir) / frame_paths[r]) for r in rows]
    out = [{'codes_max': [], 'codes_mean': []} for _ in sae_paths]
    n_fg, t0 = [], time.time()
    for b, (pix, grey, r) in enumerate(run.loader(paths, rows, batch_size, num_workers)):
        mask, pooled = run.batch(pix, grey, r.numpy())
        n_fg.append(mask.sum(1).short().cpu().numpy())
        for o, (mx, mean) in zip(out, pooled):
            o['codes_max'].append(mx.half().cpu().numpy())
            o['codes_mean'].append(mean.half().cpu().numpy())
        if b % 50 == 0:
            print(f'  batch {b:5d}  {sum(map(len, n_fg)):7d}/{len(rows)}  {sum(map(len, n_fg)) / (time.time() - t0):.1f} f/s',
                  flush=True)
    n_fg = np.concatenate(n_fg)
    return [{'codes_max': np.concatenate(o['codes_max']), 'codes_mean': np.concatenate(o['codes_mean']), 'n_fg': n_fg}
            for o in out]


def encode_fg_shard(frame_paths, lo, hi, shard_dir, sae_path, bg_dir, ann_path, dataset_dir='dataset',
                    batch_size=128, num_workers=16, device='cuda'):
    shard_dir = Path(shard_dir)
    if (shard_dir / 'DONE').exists():
        print(f'[SKIP] {shard_dir} done')
        return shard_dir
    tmp = Path(str(shard_dir) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    run = _Runner([sae_path], bg_dir, ann_path, device)
    m = run.saes[0][0].n_latents
    N = hi - lo
    rows = np.arange(lo, hi)
    paths = [str(Path(dataset_dir) / frame_paths[r]) for r in rows]
    mm = {n: np.lib.format.open_memmap(tmp / f'{n}.npy', 'w+', np.float16, (N, m)) for n in ('codes_max', 'codes_mean')}
    mm['n_fg'] = np.lib.format.open_memmap(tmp / 'n_fg.npy', 'w+', np.int16, (N,))
    t0, cur = time.time(), 0
    for b, (pix, grey, r) in enumerate(run.loader(paths, rows, batch_size, num_workers)):
        r = r.numpy()
        assert r[0] == lo + cur
        mask, [(mx, mean)] = run.batch(pix, grey, r)
        B = len(r)
        mm['codes_max'][cur:cur + B] = mx.half().cpu().numpy()
        mm['codes_mean'][cur:cur + B] = mean.half().cpu().numpy()
        mm['n_fg'][cur:cur + B] = mask.sum(1).short().cpu().numpy()
        cur += B
        if b % 100 == 0:
            print(f'  batch {b:5d}  {cur:7d}/{N}  {cur / (time.time() - t0):6.1f} frames/s', flush=True)
    assert cur == N, f'wrote {cur} rows, expected {N}'
    for a in mm.values():
        a.flush()
    del mm
    elapsed = time.time() - t0
    (tmp / 'shard.json').write_text(json.dumps({'lo': lo, 'hi': hi, 'elapsed_s': round(elapsed, 1)}))
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    tmp.rename(shard_dir)
    (shard_dir / 'DONE').touch()
    print(f'Done rows [{lo}, {hi}) in {elapsed:.0f}s ({N / elapsed:.1f} frames/s) -> {shard_dir}')
    return shard_dir


def merge_fg_shards(out_dir, ranges, n_rows):
    out_dir = Path(out_dir)
    if (out_dir / 'DONE').exists():
        print(f'[SKIP] {out_dir} already merged')
        return
    missing = [i for i in range(len(ranges)) if not (out_dir / 'shards' / f'shard_{i:02d}' / 'DONE').exists()]
    if missing:
        raise RuntimeError(f'shards not finished: {missing}')
    for name in FG_OUTPUTS:
        first = np.load(out_dir / 'shards' / 'shard_00' / f'{name}.npy', mmap_mode='r')
        dst = np.lib.format.open_memmap(out_dir / f'{name}.npy.tmp', 'w+', first.dtype, (n_rows,) + first.shape[1:])
        for i, (lo, hi) in enumerate(ranges):
            src = np.load(out_dir / 'shards' / f'shard_{i:02d}' / f'{name}.npy', mmap_mode='r')
            assert src.shape[0] == hi - lo, (name, i, src.shape)
            for a in range(0, hi - lo, 65536):
                dst[lo + a:lo + min(a + 65536, hi - lo)] = src[a:a + 65536]
        dst.flush()
        del dst
        (out_dir / f'{name}.npy.tmp').rename(out_dir / f'{name}.npy')
        print(f'  merged {name}', flush=True)
    (out_dir / 'DONE').touch()


def verify_fg_codes(out_dir, sae_path, bg_dir, ann_path, dataset_dir='dataset', n_check=64, seed=0, chunk=65536,
                    device='cuda'):
    """Row count, non-finite / all-zero rows, n_fg range, and an alignment test: n_check random
    rows recomputed from scratch (JPEG -> DINOv2 -> mask -> SAE) vs the stored rows, with a
    random-row baseline."""
    out_dir = Path(out_dir)
    res = {}
    arrs = {n: np.load(out_dir / f'{n}.npy', mmap_mode='r') for n in FG_OUTPUTS}
    for n in ('codes_max', 'codes_mean'):
        a = arrs[n]
        n_nan = n_zero = 0
        for i in range(0, a.shape[0], chunk):
            x = np.asarray(a[i:i + chunk], dtype=np.float32)
            n_nan += int((~np.isfinite(x)).any(1).sum())
            n_zero += int((x == 0).all(1).sum())
        res[n] = {'shape': list(a.shape), 'rows_nonfinite': n_nan, 'rows_all_zero': n_zero}
    nf = np.asarray(arrs['n_fg'])
    res['n_fg'] = {'shape': list(nf.shape), 'min': int(nf.min()), 'max': int(nf.max()), 'median': float(np.median(nf)),
                   'frames_without_fg': int((nf == 0).sum()),
                   'fg_frac_p5_p50_p95': [float(np.percentile(nf, q) / 1024) for q in (5, 50, 95)]}
    frame_paths = pd.read_csv(ann_path, usecols=['frame_path'])['frame_path'].values
    res['n_rows_annotations'] = len(frame_paths)
    rows = np.sort(np.random.default_rng(seed).choice(len(frame_paths), n_check, replace=False))
    [ref] = encode_rows(rows, frame_paths, [sae_path], bg_dir, ann_path, dataset_dir, batch_size=16, num_workers=8,
                        device=device)

    def cmp(a, b):
        a, b = torch.from_numpy(np.asarray(a, dtype=np.float32)), torch.from_numpy(np.asarray(b, dtype=np.float32))
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        return {'max_abs_diff': float((a - b).abs().max()), 'cos_min': float(cos.min()),
                'active_set_mismatch_frac': float(((a > 0) != (b > 0)).float().mean())}
    other = np.sort(np.random.default_rng(seed + 1).integers(0, len(frame_paths), n_check))
    res['alignment'] = {'n_frames': n_check, 'rows': rows.tolist(),
                        'codes_max': cmp(arrs['codes_max'][rows], ref['codes_max']),
                        'codes_mean': cmp(arrs['codes_mean'][rows], ref['codes_mean']),
                        'n_fg_exact_match_frac': float((nf[rows] == ref['n_fg']).mean()),
                        'n_fg_max_abs_diff': int(np.abs(nf[rows].astype(int) - ref['n_fg']).max()),
                        'baseline_random_rows_codes_max': cmp(arrs['codes_max'][other], ref['codes_max'])}
    return res
