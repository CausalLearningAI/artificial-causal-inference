"""
Fused foreground encoding (ECI): ONE DINOv2 forward per frame gives the pooled SAE codes of
src/eci/fg_encode.py (codes_max, codes_mean, n_fg) AND the SOMP codes of src/eci/somp_encode.py
(codes_somp, somp_idx, energy, n_fg), instead of two separate passes over all frames.

Both outputs are written in exactly the shard layout of the two separate passes, so the existing
merge / verify code (scripts/eci/fg_encode_all.py --merge --verify, scripts/eci/somp_encode_all.py
--merge --verify, --link-mean) runs unchanged on them:
    <codes>/<sae>/shards/shard_XX/        codes_max, codes_mean, n_fg, shard.json, DONE
    <codes>/<sae>_somp/shards/shard_XX/   codes_somp, somp_idx, energy, n_fg, shard.json, DONE

Per batch, exactly the computations of the two passes on the same tokens:
    tokens, mask  SompEncoder.tokens = encode_batch(DINOv2 448) + the SAE's foreground rule
                  (the same calls as fg_encode._Runner.batch for a static dinov2_base SAE)
    pooled codes  fg_encode.fg_sae_pool(sae, norm, tokens, mask)
    SOMP          SompEncoder.batch(tokens, mask)  (K atoms, x = norm(token) - b_dec, dictionary W_dec)
Supported: static-token dinov2_base SAEs without background subtraction (as somp_encode).
The SOMP shard.json 'codes_max_alignment_cos' compares the SOMP pass's own SAE max pool with the pooled
codes of the same pass (same tokens; there is no stored reference yet), so it is a consistency check only.

Functions:
    FusedEncoder        SompEncoder + the pooled-codes head on the same tokens
    encode_fused_shard  rows [lo, hi) -> both shard directories
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from src.eci.fg_encode import fg_sae_pool
from src.eci.somp_encode import SompEncoder, _cos


class FusedEncoder(SompEncoder):
    def __init__(self, sae_path, K, frame_paths, dataset_dir, bg_dir, ann_path, device='cuda'):
        super().__init__('fg', sae_path, K, frame_paths, dataset_dir, bg_dir, ann_path, device)
        if self.run.bg_sub:
            raise ValueError('fused encoding does not support background-subtracted SAEs (bg_sub)')

    @torch.no_grad()
    def fused_batch(self, item, rows):
        """item = loader batch (pix, grey, rows) -> dict: codes_max / codes_mean (pooled, B x m float32),
        n_fg, and the SOMP outputs codes_somp / somp_idx / energy / somp_codes_max."""
        tok, mask = self.tokens(item, rows)
        mx, mean = fg_sae_pool(self.sae, self.norm, tok, mask)
        o = self.batch(tok, mask)
        return {'codes_max': mx, 'codes_mean': mean, 'n_fg': mask.sum(1), 'codes_somp': o['codes_somp'],
                'somp_idx': o['somp_idx'], 'energy': o['energy'], 'somp_codes_max': o['codes_max']}


def _fresh(shard_dir):
    tmp = Path(str(shard_dir) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    return tmp


def _finish(tmp, shard_dir):
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    tmp.rename(shard_dir)
    (shard_dir / 'DONE').touch()


def encode_fused_shard(enc, lo, hi, fg_shard_dir, somp_shard_dir, batch_size=128, num_workers=16):
    """Rows [lo, hi) -> fg_shard_dir (codes_max, codes_mean, n_fg) and somp_shard_dir (codes_somp, somp_idx,
    energy, n_fg). Skipped when both are DONE; otherwise both are (re)written."""
    fg_shard_dir, somp_shard_dir = Path(fg_shard_dir), Path(somp_shard_dir)
    if (fg_shard_dir / 'DONE').exists() and (somp_shard_dir / 'DONE').exists():
        print(f'[SKIP] {fg_shard_dir} and {somp_shard_dir} done')
        return
    tf, ts = _fresh(fg_shard_dir), _fresh(somp_shard_dir)
    N, K, m = hi - lo, enc.K, enc.m
    mf = {n: np.lib.format.open_memmap(tf / f'{n}.npy', 'w+', np.float16, (N, m)) for n in ('codes_max', 'codes_mean')}
    mf['n_fg'] = np.lib.format.open_memmap(tf / 'n_fg.npy', 'w+', np.int16, (N,))
    ms = {'codes_somp': np.lib.format.open_memmap(ts / 'codes_somp.npy', 'w+', np.float16, (N, m)),
          'somp_idx': np.lib.format.open_memmap(ts / 'somp_idx.npy', 'w+', np.int16, (N, K)),
          'energy': np.lib.format.open_memmap(ts / 'energy.npy', 'w+', np.float32, (N, 2 + K)),
          'n_fg': np.lib.format.open_memmap(ts / 'n_fg.npy', 'w+', np.int16, (N,))}
    rows = np.arange(lo, hi)
    cos_all, t0, cur = [], time.time(), 0
    for b, item in enumerate(enc.loader(rows, batch_size, num_workers)):
        r = item[2].numpy()
        assert r[0] == lo + cur
        B = len(r)
        o = enc.fused_batch(item, r)
        cos_all.append(_cos(o['somp_codes_max'].half(), o['codes_max'].half()).cpu())
        nf = o['n_fg'].short().cpu().numpy()
        mf['codes_max'][cur:cur + B] = o['codes_max'].half().cpu().numpy()
        mf['codes_mean'][cur:cur + B] = o['codes_mean'].half().cpu().numpy()
        mf['n_fg'][cur:cur + B] = nf
        ms['codes_somp'][cur:cur + B] = o['codes_somp'].half().cpu().numpy()
        ms['somp_idx'][cur:cur + B] = o['somp_idx'].short().cpu().numpy()
        ms['energy'][cur:cur + B] = o['energy'].cpu().numpy()
        ms['n_fg'][cur:cur + B] = nf
        cur += B
        if b % 100 == 0:
            c = torch.cat(cos_all)
            print(f'  batch {b:5d}  {cur:7d}/{N}  {cur / (time.time() - t0):6.1f} frames/s  '
                  f'SOMP-pass vs pooled codes_max cos min {float(c.min()):.4f}', flush=True)
        if float(cos_all[-1].median()) < 0.99:
            raise RuntimeError(f'batch {b}: SOMP pass and pooled codes disagree (cos median '
                               f'{float(cos_all[-1].median()):.4f})')
    assert cur == N, f'wrote {cur} rows, expected {N}'
    for a in list(mf.values()) + list(ms.values()):
        a.flush()
    del mf, ms
    c = torch.cat(cos_all).numpy()
    elapsed = time.time() - t0
    (tf / 'shard.json').write_text(json.dumps({'lo': lo, 'hi': hi, 'elapsed_s': round(elapsed, 1), 'fused_with_somp': True}))
    (ts / 'shard.json').write_text(json.dumps({
        'lo': lo, 'hi': hi, 'elapsed_s': round(elapsed, 1), 'fused_with_codes': True,
        'codes_max_alignment_ref': 'pooled codes_max of the same fused pass (same tokens)',
        'codes_max_alignment_cos': {'min': float(c.min()), 'p1': float(np.percentile(c, 1)), 'median': float(np.median(c)),
                                    'frac_below_0.99': float((c < 0.99).mean())}}))
    _finish(tf, fg_shard_dir)
    _finish(ts, somp_shard_dir)
    print(f'Done rows [{lo}, {hi}) in {elapsed:.0f}s ({N / elapsed:.1f} frames/s) -> {fg_shard_dir} + {somp_shard_dir}')
