"""
Encode every mice v1 frame with DINOv2 + the ECI sparse autoencoder (ECI step 3).

For each frame: final-layer DINOv2 patch tokens (same model and preprocessing as
src/eci/extract.py) -> rounded to float16 exactly as the SAE training tokens were ->
SAE codes per patch (global inference threshold) -> pooled over the patches:
    codes_mean  mean over patches   (N, n_latents) float16
    codes_max   max over patches    (N, n_latents) float16  (small localized concepts)
    cls_l-1     CLS token, same layer, (N, 768) float16

Work is sharded by row ranges of annotations.csv; each shard writes its own files
into shards/shard_XX.tmp/ and renames to shards/shard_XX/ + a DONE flag when complete,
so a job array can be resubmitted and finished shards are skipped. merge_shards()
concatenates them into the final arrays.

Functions:
    shard_ranges     row ranges [lo, hi) for n_shards
    encode_shard     DINOv2 + SAE on rows [lo, hi) -> shard directory
    merge_shards     shard files -> final (N, .) memmaps + DONE flag
    verify_codes     row count, NaN / all-zero rows, alignment with step-1 stored tokens
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eci.extract import _FrameDataset, load_encoder
from src.eci.sae import load_sae
from src.eci.token_store import TokenStore

OUTPUTS = ('codes_mean', 'codes_max', 'cls_l-1')


def shard_ranges(n_rows, n_shards):
    edges = np.linspace(0, n_rows, n_shards + 1).round().astype(np.int64)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


@torch.no_grad()
def sae_pool(sae, norm, patch_tokens_fp16):
    """(B, P, d) float16 tokens -> mean- and max-pooled SAE codes (B, m) float32."""
    B, P, d = patch_tokens_fp16.shape
    z = sae.encode(norm(patch_tokens_fp16.reshape(B * P, d)), mode='threshold').view(B, P, -1)
    return z.mean(1), z.amax(1)


def encode_shard(frame_paths, lo, hi, shard_dir, sae_path, dataset_dir='dataset', encoder='dinov2_base',
                 resolution=224, batch_size=256, num_workers=16, device='cuda', outputs=OUTPUTS):
    shard_dir = Path(shard_dir)
    if (shard_dir / 'DONE').exists():
        print(f'[SKIP] {shard_dir} done')
        return shard_dir
    tmp = Path(str(shard_dir) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    device = torch.device(device)
    _, processor, model = load_encoder(encoder, resolution, device)
    sae, norm, _ = load_sae(sae_path, device)
    n_prefix = 1 + (getattr(model.config, 'num_register_tokens', 0) or 0)
    n_patches = (resolution // model.config.patch_size) ** 2
    dim, m = model.config.hidden_size, sae.n_latents
    N = hi - lo

    paths = [str(Path(dataset_dir) / p) for p in frame_paths[lo:hi]]
    loader = torch.utils.data.DataLoader(
        _FrameDataset(paths, processor), batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=device.type == 'cuda', prefetch_factor=4 if num_workers > 0 else None)
    width = {'codes_mean': m, 'codes_max': m, 'cls_l-1': dim}
    mm = {n: np.lib.format.open_memmap(tmp / f'{n}.npy', 'w+', np.float16, (N, width[n])) for n in outputs}

    t0, cur = time.time(), 0
    for b, pix in enumerate(loader):
        with torch.inference_mode():
            hs = model(pixel_values=pix.to(device, non_blocking=True)).last_hidden_state.float()
            if hs.shape[1] != n_prefix + n_patches or not torch.isfinite(hs).all():
                raise RuntimeError(f'bad DINOv2 output in batch {b}: shape {tuple(hs.shape)}')
            hs = hs.half()  # identical rounding to the stored SAE training tokens
            mean, mx = sae_pool(sae, norm, hs[:, n_prefix:])
        B = hs.shape[0]
        vals = {'codes_mean': mean, 'codes_max': mx, 'cls_l-1': hs[:, 0]}
        for n in outputs:
            mm[n][cur:cur + B] = vals[n].half().cpu().numpy()
        cur += B
        if b % 50 == 0:
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


def merge_shards(out_dir, ranges, n_rows, outputs=OUTPUTS):
    out_dir = Path(out_dir)
    if (out_dir / 'DONE').exists():
        print(f'[SKIP] {out_dir} already merged')
        return
    missing = [i for i in range(len(ranges)) if not (out_dir / 'shards' / f'shard_{i:02d}' / 'DONE').exists()]
    if missing:
        raise RuntimeError(f'shards not finished: {missing}')
    for name in outputs:
        first = np.load(out_dir / 'shards' / 'shard_00' / f'{name}.npy', mmap_mode='r')
        dst = np.lib.format.open_memmap(out_dir / f'{name}.npy.tmp', 'w+', np.float16, (n_rows, first.shape[1]))
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


@torch.no_grad()
def verify_codes(out_dir, sae_path, tokens_dir, n_check=50, seed=0, chunk=65536, outputs=OUTPUTS):
    """Returns a dict of checks. The alignment test re-encodes n_check training-sample
    frames from the step-1 stored patch tokens (either TokenStore layout) and compares
    to the rows given by row_idx."""
    out_dir, tokens_dir = Path(out_dir), Path(tokens_dir)
    res = {}
    arrs = {n: np.load(out_dir / f'{n}.npy', mmap_mode='r') for n in outputs}
    for n, a in arrs.items():
        n_nan = n_zero = 0
        for i in range(0, a.shape[0], chunk):
            x = np.asarray(a[i:i + chunk], dtype=np.float32)
            n_nan += int((~np.isfinite(x)).any(1).sum())
            n_zero += int((x == 0).all(1).sum())
        res[n] = {'shape': list(a.shape), 'rows_nonfinite': n_nan, 'rows_all_zero': n_zero}

    sae, norm, _ = load_sae(sae_path, 'cpu')
    store = TokenStore(tokens_dir)
    meta = store.meta[store.meta.row_idx >= 0]
    pick = np.sort(np.random.default_rng(seed).choice(meta.index.values, n_check, replace=False))
    rows = meta.loc[pick, 'row_idx'].values
    ref_mean, ref_max = sae_pool(sae, norm, torch.from_numpy(store.frames(pick)))
    got_mean = torch.from_numpy(np.asarray(arrs['codes_mean'][rows], dtype=np.float32))
    got_max = torch.from_numpy(np.asarray(arrs['codes_max'][rows], dtype=np.float32))

    def cmp(a, b):
        a, b = a.float(), b.float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        return {'max_abs_diff': float((a - b).abs().max()), 'rel_l2_err_max': float(((a - b).norm(dim=1) / b.norm(dim=1)).max()),
                'cos_min': float(cos.min()), 'active_set_mismatch_frac': float(((a > 0) != (b > 0)).float().mean())}
    # a random-row baseline shows what a misaligned row would look like
    other = np.random.default_rng(seed + 1).integers(0, arrs['codes_mean'].shape[0], n_check)
    res['alignment'] = {
        'n_frames': n_check, 'tokens_dir': str(tokens_dir),
        'codes_mean': cmp(got_mean, ref_mean.half()),
        'codes_max': cmp(got_max, ref_max.half()),
        'baseline_random_rows_codes_mean': cmp(torch.from_numpy(np.asarray(arrs['codes_mean'][np.sort(other)], dtype=np.float32)),
                                              ref_mean.half()),
    }
    if 'cls_l-1' in arrs:
        got_cls = np.asarray(arrs['cls_l-1'][rows], dtype=np.float32)
        res['alignment']['cls'] = cmp(torch.from_numpy(got_cls), torch.from_numpy(store.cls(pick).astype(np.float32)))
    return res
