"""
Encode mice v1 frames with DINOv2 (whole frame, 448, no crop) + foreground mask + one or
more foreground SAEs (ECI, "mouse patches" pipeline, src/eci/foreground.py).

Per frame and SAE, pooled over the FOREGROUND patches only:
    codes_max   max over foreground patches    (N, n_latents) float16 (0 if no foreground)
    codes_mean  mean over foreground patches   (N, n_latents) float16 (0 if no foreground)
    n_fg        number of foreground patches   (N,) int16
Sharded by row ranges of annotations.csv, same resumable layout as src/eci/encode.py
(shards/shard_XX.tmp -> shards/shard_XX + DONE; merge_fg_shards; verify_fg_codes).

The foreground rule and the SAE input come from the SAE checkpoint: 'fg_rule' (src/eci/foreground.py
RULES, default 'fg448' for checkpoints without it) and 'motion_delta' D (0 = static token; D > 0 =
[token_t, token_t - token_{t-D}] at the same patch, the frame D rows earlier in the same video, clipped
to the video's first frame). All SAEs of one run must share the rule. 'bg_sub' (absent = False): the SAE input is
token - the video's background token at the same patch and time block (FgBackgrounds.background); the mask is
computed from the raw tokens as always. 'encoder' (absent = 'dinov2_base') is the
encoder of the SAE input tokens (src/eci/foreground.py FgEncoder: the mask always comes from DINOv2 at 448; a
'dinov3_base' SAE encodes DINOv3 tokens of the same patches); all SAEs of one run must share it too. Frames earlier in the same batch
or the previous batch are reused; any other earlier frame is loaded and encoded on the fly. 'align' (absent = 'none')
is the frame alignment of the SAE's training tokens (src/eci/foreground.py align_rot90: 'odor' rotates every frame so
the odor corner is at the top right); frames are rotated the same way here, and the backgrounds must record it.

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

from src.eci.foreground import (MASK_ENCODER, RULES, FgBackgrounds, FgEncoder, FrameDatasetFG, align_rot90, encode_batch,
                                obs_rows, subtract_background_tokens)
from src.eci.sae import load_sae

FG_OUTPUTS = ('codes_max', 'codes_mean', 'n_fg')


@torch.no_grad()
def fg_sae_pool(sae, norm, tokens, mask, prev=None):
    """tokens (B, P, d) fp16, mask (B, P) bool -> (max (B, m), mean (B, m)) float32 over the
    masked patches (zeros for frames without foreground). Only masked tokens are encoded.
    prev (B, P, d) fp16: tokens D frames earlier; the SAE input is then [token, token - prev]."""
    B, P, _ = tokens.shape
    fi, pi = torch.nonzero(mask, as_tuple=True)
    x = tokens[fi, pi]
    if prev is not None:
        x = torch.cat([x, (x.float() - prev[fi, pi].float()).half()], 1)
    z = sae.encode(norm(x), mode='threshold')  # (n, m), >= 0
    m = z.shape[1]
    mx = torch.zeros(B, m, device=z.device).index_reduce_(0, fi, z, 'amax', include_self=True)
    sm = torch.zeros(B, m, device=z.device).index_add_(0, fi, z)
    n = mask.sum(1, keepdim=True).float()
    return mx, sm / n.clamp_min(1)


class _Runner:
    def __init__(self, sae_paths, bg_dir, ann_path, device='cuda', frame_paths=None, dataset_dir='dataset'):
        self.device = torch.device(device)
        loaded = [load_sae(p, self.device) for p in sae_paths]
        encoders = {ck.get('encoder', MASK_ENCODER) for _, _, ck in loaded}
        if len(encoders) != 1:
            raise ValueError(f'SAEs with different encoders in one run: {encoders}')
        self.encoder = encoders.pop()
        self.fg = FgEncoder(self.encoder, self.device)
        self.processor, self.model = self.fg.processor, self.fg.model
        self.saes = [(s, n) for s, n, _ in loaded]
        self.deltas = [int(ck.get('motion_delta', 0) or 0) for _, _, ck in loaded]
        if self.fg.model2 is not None and max(self.deltas) > 0:
            raise ValueError(f'motion inputs are only supported for {MASK_ENCODER} SAEs')
        bg_subs = {bool(ck.get('bg_sub', False)) for _, _, ck in loaded}
        if len(bg_subs) != 1:
            raise ValueError('SAEs with and without background subtraction in one run')
        self.bg_sub = bg_subs.pop()
        if self.bg_sub and (self.fg.model2 is not None or max(self.deltas) > 0):
            raise ValueError(f'background-subtracted inputs are only supported for static {MASK_ENCODER} SAEs')
        rules = {ck.get('fg_rule', 'fg448') for _, _, ck in loaded}
        if len(rules) != 1:
            raise ValueError(f'SAEs with different foreground rules in one run: {rules}')
        self.rule_name = rules.pop()
        self.rule = RULES[self.rule_name]
        aligns = {ck.get('align', 'none') for _, _, ck in loaded}
        if len(aligns) != 1:
            raise ValueError(f'SAEs with different frame alignments in one run: {aligns}')
        self.align = aligns.pop()
        self.rot = align_rot90(self.align, ann_path)  # None or per-row 90-degree turns
        ranges = obs_rows(ann_path)
        self.bgs = FgBackgrounds(bg_dir, ranges, self.rule, self.device, align=self.align)
        self.frame_paths, self.dataset_dir = frame_paths, Path(dataset_dir)
        self._cache = {}  # row -> tokens (1024, d) of the previous batch (for motion inputs)

    @torch.no_grad()
    def _prev_tokens(self, tok, rows, D):
        """(B, 1024, d) tokens of rows r - D (same video, clipped to its first row)."""
        k = self.bgs.obs_index(rows)
        prow = np.maximum(rows - D, self.bgs.starts[k])
        where = {int(r): i for i, r in enumerate(rows)}
        out = torch.empty_like(tok)
        missing = []
        for i, pr in enumerate(prow):
            pr = int(pr)
            if pr in where:
                out[i] = tok[where[pr]]
            elif pr in self._cache:
                out[i] = self._cache[pr]
            else:
                missing.append((i, pr))
        if missing:
            ds = FrameDatasetFG([str(self.dataset_dir / self.frame_paths[pr]) for _, pr in missing], self.processor,
                                rot=self.rot_of([pr for _, pr in missing]))
            pix = torch.stack([ds[j][0] for j in range(len(ds))])
            for a in range(0, len(missing), 64):
                enc = encode_batch(self.model, pix[a:a + 64], self.device)
                for (i, _), t in zip(missing[a:a + 64], enc):
                    out[i] = t
        return out

    def rot_of(self, rows):
        """Per-row 90-degree turns of the frames (None when not aligned)."""
        return None if self.rot is None else self.rot[np.asarray(rows, dtype=np.int64)]

    def loader(self, paths, rows, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            self.fg.dataset(paths, rows, self.rot_of(rows)), batch_size=batch_size, num_workers=num_workers,
            shuffle=False, pin_memory=self.device.type == 'cuda', prefetch_factor=4 if num_workers > 0 else None)

    def batch(self, pix, grey, rows, pix2=None):
        """pix2: the second encoder's pixel_values (loader batch[3]) for a non-DINOv2 SAE."""
        rows = np.asarray(rows)
        tok = encode_batch(self.model, pix, self.device)
        mask, _ = self.bgs.mask(tok, grey.to(self.device, non_blocking=True), rows)
        tok = self.fg.sae_tokens(tok, pix2)
        if self.bg_sub:  # SAE input = token - the video's background token (same position, time block)
            tok = subtract_background_tokens(tok, self.bgs.background(rows))
        prev = {D: self._prev_tokens(tok, rows, D) for D in set(self.deltas) if D > 0}
        out = [fg_sae_pool(s, n, tok, mask, prev.get(D)) for (s, n), D in zip(self.saes, self.deltas)]
        Dm = max(self.deltas)
        if Dm > 0:  # keep the last Dm rows for the next batch
            self._cache = {int(r): tok[i] for i, r in enumerate(rows[-Dm:], start=len(rows) - min(Dm, len(rows)))}
        return mask, out


def encode_rows(rows, frame_paths, sae_paths, bg_dir, ann_path, dataset_dir='dataset', batch_size=128,
                num_workers=16, device='cuda'):
    """-> list (one per SAE) of dicts codes_max / codes_mean (N, m) float16, plus n_fg (N,)."""
    run = _Runner(sae_paths, bg_dir, ann_path, device, frame_paths, dataset_dir)
    rows = np.asarray(rows)
    paths = [str(Path(dataset_dir) / frame_paths[r]) for r in rows]
    out = [{'codes_max': [], 'codes_mean': []} for _ in sae_paths]
    n_fg, t0 = [], time.time()
    for b, (pix, grey, r, *pix2) in enumerate(run.loader(paths, rows, batch_size, num_workers)):
        mask, pooled = run.batch(pix, grey, r.numpy(), *pix2)
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
    run = _Runner([sae_path], bg_dir, ann_path, device, frame_paths, dataset_dir)
    m = run.saes[0][0].n_latents
    N = hi - lo
    rows = np.arange(lo, hi)
    paths = [str(Path(dataset_dir) / frame_paths[r]) for r in rows]
    mm = {n: np.lib.format.open_memmap(tmp / f'{n}.npy', 'w+', np.float16, (N, m)) for n in ('codes_max', 'codes_mean')}
    mm['n_fg'] = np.lib.format.open_memmap(tmp / 'n_fg.npy', 'w+', np.int16, (N,))
    t0, cur = time.time(), 0
    for b, (pix, grey, r, *pix2) in enumerate(run.loader(paths, rows, batch_size, num_workers)):
        r = r.numpy()
        assert r[0] == lo + cur
        mask, [(mx, mean)] = run.batch(pix, grey, r, *pix2)
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


def merge_fg_shards(out_dir, ranges, n_rows, names=FG_OUTPUTS):
    out_dir = Path(out_dir)
    if (out_dir / 'DONE').exists():
        print(f'[SKIP] {out_dir} already merged')
        return
    missing = [i for i in range(len(ranges)) if not (out_dir / 'shards' / f'shard_{i:02d}' / 'DONE').exists()]
    if missing:
        raise RuntimeError(f'shards not finished: {missing}')
    for name in names:
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
