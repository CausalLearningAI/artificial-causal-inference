"""
SOMP frame codes for every frame (ECI): DINOv2 tokens -> the SAE's token set of the frame -> SOMP
(src/eci/somp.py) over the SAE decoder -> one sparse vector per frame.

The tokens and the token set follow the codes the SAE already has (<codes>/<sae>/config.json):
    'fg'    foreground pipeline (src/eci/foreground.py, src/eci/fg_encode.py): whole frame at 448,
            the SAE's foreground rule; SOMP over the foreground patches
    'full'  full-frame pipeline (src/eci/encode.py): 224 center crop, all 256 patches
Tokens are rounded to float16 exactly as for the SAE codes, then x = norm(token) - b_dec.

Per frame (row of annotations.csv) the shard / final folder holds:
    codes_somp  float16 (N, m)      SOMP importance of each chosen atom (RMS over the frame's tokens of
                                    its least-squares coefficient), 0 for the other atoms
    somp_idx    int16   (N, K)      chosen atoms in selection order (-1 = not needed, frame explained)
    energy      float32 (N, 2 + K)  [sum_t ||x_t||^2, SAE residual energy sum_t ||x_t - z_t W_dec||^2 (the SAE's
                                    own per-token reconstruction, threshold inference), SOMP residual energy
                                    after 1..K atoms]
    n_fg        int16   (N,)        tokens per frame ('fg' only)
Every batch also recomputes the SAE max-pooled codes from the same tokens and compares them with the
stored <codes>/<sae>/codes_max.npy rows (alignment; cosine per frame, summarised in shard.json).
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from src.eci.somp import sae_dictionary, somp_batched

SOMP_OUTPUTS = ('codes_somp', 'somp_idx', 'energy')


class SompEncoder:
    def __init__(self, pipeline, sae_path, K, frame_paths, dataset_dir, bg_dir=None, ann_path=None, device='cuda'):
        self.pipeline, self.K, self.device = pipeline, K, torch.device(device)
        self.frame_paths, self.dataset_dir = frame_paths, Path(dataset_dir)
        if pipeline == 'fg':
            from src.eci.fg_encode import _Runner
            self.run = _Runner([sae_path], bg_dir, ann_path, device, frame_paths, dataset_dir)
            if self.run.deltas != [0]:
                raise ValueError('SOMP encoding supports static-token SAEs only (motion_delta 0)')
            if self.run.fg.model2 is not None:
                raise ValueError(f'SOMP encoding supports dinov2_base SAEs only (this one: {self.run.encoder})')
            self.sae, self.norm = self.run.saes[0]
            self.processor, self.model = self.run.processor, self.run.model
        elif pipeline == 'full':
            from src.eci.extract import load_encoder
            from src.eci.sae import load_sae
            _, self.processor, self.model = load_encoder('dinov2_base', 224, self.device)
            self.sae, self.norm, _ = load_sae(sae_path, self.device)
        else:
            raise ValueError(pipeline)
        self.to_x, self.D = sae_dictionary(self.sae, self.norm)
        self.gram = self.D.double() @ self.D.double().T
        self.m = self.sae.n_latents

    def loader(self, rows, batch_size, num_workers):
        paths = [str(self.dataset_dir / self.frame_paths[r]) for r in rows]
        if self.pipeline == 'fg':
            return self.run.loader(paths, np.asarray(rows), batch_size, num_workers)
        from src.eci.extract import _FrameDataset
        return torch.utils.data.DataLoader(
            _FrameDataset(paths, self.processor), batch_size=batch_size, num_workers=num_workers, shuffle=False,
            pin_memory=self.device.type == 'cuda', prefetch_factor=4 if num_workers > 0 else None)

    @torch.no_grad()
    def tokens(self, item, rows):
        """-> tokens (B, P, d) float16, mask (B, P) bool."""
        if self.pipeline == 'fg':
            from src.eci.foreground import encode_batch
            pix, grey = item[0], item[1]
            tok = encode_batch(self.model, pix, self.device)
            mask, _ = self.run.bgs.mask(tok, grey.to(self.device, non_blocking=True), np.asarray(rows))
            return tok, mask
        n_prefix = 1 + (getattr(self.model.config, 'num_register_tokens', 0) or 0)
        with torch.inference_mode():
            hs = self.model(pixel_values=item.to(self.device, non_blocking=True)).last_hidden_state.float()
        if hs.shape[1] != n_prefix + 256 or not torch.isfinite(hs).all():
            raise RuntimeError(f'bad DINOv2 output: shape {tuple(hs.shape)}')
        tok = hs[:, n_prefix:].half()
        return tok, torch.ones(tok.shape[:2], dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def batch(self, tok, mask):
        """-> dict of float/ int tensors for the batch (see module docstring) + 'codes_max' (SAE max pool)."""
        B, P, d = tok.shape
        n = mask.sum(1)
        nmax = max(int(n.max()), 1)
        order = torch.argsort(mask.to(torch.int8), dim=1, descending=True, stable=True)[:, :nmax]  # fg first
        sel = tok.gather(1, order[..., None].expand(B, nmax, d))
        pm = torch.arange(nmax, device=tok.device)[None] < n[:, None]
        X = self.to_x(sel.reshape(-1, d)).view(B, nmax, d) * pm[..., None]
        # SAE's own reconstruction and max pool on the same tokens
        fi, pi = torch.nonzero(pm, as_tuple=True)
        z = self.sae.encode(self.norm(sel[fi, pi]), mode='threshold')
        xs = X[fi, pi]
        e_sae = torch.zeros(B, dtype=torch.float64, device=tok.device).index_add_(
            0, fi, (xs - z @ self.D).double().pow(2).sum(1))
        cmax = torch.zeros(B, self.m, device=tok.device).index_reduce_(0, fi, z, 'amax', include_self=True)
        o = somp_batched(X, pm, self.D, self.K, gram=self.gram)
        codes = torch.zeros(B, self.m, dtype=torch.float64, device=tok.device)
        codes.scatter_add_(1, o['idx'].clamp_min(0), o['importance'] * (o['idx'] >= 0))
        energy = torch.cat([o['energy'][:, None], e_sae[:, None], o['resid']], 1)
        return {'codes_somp': codes.float(), 'somp_idx': o['idx'], 'energy': energy.float(), 'n_fg': n,
                'codes_max': cmax}


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1, eps=1e-8)


def _cos_fg(a, b):
    """_cos, with 1 for rows that are zero in both (a frame with an empty foreground mask has zero codes in both
    passes: agreement, not a misalignment; tadpole frames where the tadpole hides under the dish rim). Any run whose
    checks passed with _cos had no such row, so its checks are unchanged."""
    both0 = (a.float().abs().sum(1) == 0) & (b.float().abs().sum(1) == 0)
    return torch.where(both0, torch.ones_like(both0, dtype=torch.float32), _cos(a, b))


def encode_somp_shard(enc, lo, hi, shard_dir, ref_codes_max, batch_size=128, num_workers=16):
    """Rows [lo, hi) -> shard_dir (memmaps + shard.json + DONE); resumable like encode_fg_shard."""
    shard_dir = Path(shard_dir)
    if (shard_dir / 'DONE').exists():
        print(f'[SKIP] {shard_dir} done')
        return shard_dir
    tmp = Path(str(shard_dir) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    N, K, m = hi - lo, enc.K, enc.m
    mm = {'codes_somp': np.lib.format.open_memmap(tmp / 'codes_somp.npy', 'w+', np.float16, (N, m)),
          'somp_idx': np.lib.format.open_memmap(tmp / 'somp_idx.npy', 'w+', np.int16, (N, K)),
          'energy': np.lib.format.open_memmap(tmp / 'energy.npy', 'w+', np.float32, (N, 2 + K))}
    if enc.pipeline == 'fg':
        mm['n_fg'] = np.lib.format.open_memmap(tmp / 'n_fg.npy', 'w+', np.int16, (N,))
    ref = np.load(ref_codes_max, mmap_mode='r')
    rows = np.arange(lo, hi)
    cos_all, t0, cur = [], time.time(), 0
    for b, item in enumerate(enc.loader(rows, batch_size, num_workers)):
        if enc.pipeline == 'fg':
            assert int(item[2][0]) == lo + cur
            Bn = len(item[2])
        else:
            Bn = item.shape[0]
        r = rows[cur:cur + Bn]
        tok, mask = enc.tokens(item, r)
        out = enc.batch(tok, mask)
        cos_all.append(_cos(out['codes_max'].half(), torch.from_numpy(np.asarray(ref[lo + cur:lo + cur + Bn])).to(tok.device)).cpu())
        mm['codes_somp'][cur:cur + Bn] = out['codes_somp'].half().cpu().numpy()
        mm['somp_idx'][cur:cur + Bn] = out['somp_idx'].short().cpu().numpy()
        mm['energy'][cur:cur + Bn] = out['energy'].cpu().numpy()
        if 'n_fg' in mm:
            mm['n_fg'][cur:cur + Bn] = out['n_fg'].short().cpu().numpy()
        cur += Bn
        if b % 100 == 0:
            c = torch.cat(cos_all)
            print(f'  batch {b:5d}  {cur:7d}/{N}  {cur / (time.time() - t0):6.1f} frames/s  '
                  f'codes_max alignment cos min {float(c.min()):.4f} median {float(c.median()):.5f}', flush=True)
        # a frame whose foreground mask differs by one border patch (GPU numerics) can reach cos ~0.95;
        # misaligned rows give cos ~0.5 (verify.json baselines), so the batch median is the alignment test
        if float(cos_all[-1].median()) < 0.99:
            raise RuntimeError(f'batch {b}: recomputed codes_max does not match the stored rows '
                               f'(cos median {float(cos_all[-1].median()):.4f}): tokens misaligned')
    assert cur == N, f'wrote {cur} rows, expected {N}'
    for a in mm.values():
        a.flush()
    del mm
    c = torch.cat(cos_all).numpy()
    elapsed = time.time() - t0
    (tmp / 'shard.json').write_text(json.dumps({
        'lo': lo, 'hi': hi, 'elapsed_s': round(elapsed, 1),
        'codes_max_alignment_cos': {'min': float(c.min()), 'p1': float(np.percentile(c, 1)), 'median': float(np.median(c)),
                                    'frac_below_0.99': float((c < 0.99).mean())}}))
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    tmp.rename(shard_dir)
    (shard_dir / 'DONE').touch()
    print(f'Done rows [{lo}, {hi}) in {elapsed:.0f}s ({N / elapsed:.1f} frames/s) -> {shard_dir}')
    return shard_dir


def verify_somp(out_dir, enc, n_rows, ranges, n_check=64, seed=0, chunk=65536):
    """Row counts, non-finite rows, atoms per frame, alignment (per-shard codes_max cosines) and a
    from-scratch recomputation of n_check random rows (JPEG -> DINOv2 -> token set -> SAE -> SOMP)."""
    out_dir = Path(out_dir)
    res = {'n_rows_annotations': n_rows}
    cs = np.load(out_dir / 'codes_somp.npy', mmap_mode='r')
    idx = np.load(out_dir / 'somp_idx.npy', mmap_mode='r')
    en = np.load(out_dir / 'energy.npy', mmap_mode='r')
    n_nan = 0
    nnz = []
    for i in range(0, cs.shape[0], chunk):
        x = np.asarray(cs[i:i + chunk], dtype=np.float32)
        n_nan += int((~np.isfinite(x)).any(1).sum())
        nnz.append((x > 0).sum(1))
    nnz = np.concatenate(nnz)
    na = (np.asarray(idx) >= 0).sum(1)
    res['codes_somp'] = {'shape': list(cs.shape), 'rows_nonfinite': n_nan,
                         'nonzero_per_frame': {'min': int(nnz.min()), 'median': float(np.median(nnz)), 'max': int(nnz.max())},
                         'atoms_used_per_frame': {'min': int(na.min()), 'median': float(np.median(na)),
                                                  'frac_frames_all_K': float((na == idx.shape[1]).mean())}}
    E = np.asarray(en, dtype=np.float64)
    res['energy_explained'] = {'sae_pooled': float(1 - E[:, 1].sum() / E[:, 0].sum()),
                               'somp_pooled_by_K': [float(1 - E[:, 2 + k].sum() / E[:, 0].sum()) for k in range(idx.shape[1])],
                               'sae_frame_median': float(np.median(1 - E[:, 1] / E[:, 0])),
                               'somp_frame_median': float(np.median(1 - E[:, -1] / E[:, 0]))}
    al = [json.loads((out_dir / 'shards' / f'shard_{i:02d}' / 'shard.json').read_text())['codes_max_alignment_cos']
          for i in range(len(ranges))] if (out_dir / 'shards').exists() else []
    res['codes_max_alignment_cos_min'] = min(a['min'] for a in al) if al else None
    res['codes_max_alignment_cos_median_min'] = min(a['median'] for a in al) if al else None
    res['codes_max_alignment_frac_below_0.99_max'] = max(a['frac_below_0.99'] for a in al) if al else None
    rows = np.sort(np.random.default_rng(seed).choice(n_rows, n_check, replace=False))
    got = {'codes_somp': [], 'somp_idx': []}
    for item in enc.loader(rows, 16, 8):
        tok, mask = enc.tokens(item, item[2].numpy() if enc.pipeline == 'fg' else None)
        o = enc.batch(tok, mask)
        got['codes_somp'].append(o['codes_somp'].half().cpu())
        got['somp_idx'].append(o['somp_idx'].short().cpu())
    ref_c = torch.cat(got['codes_somp'])
    ref_i = torch.cat(got['somp_idx']).numpy()
    st_c = torch.from_numpy(np.asarray(cs[rows]))
    cos = _cos_fg(st_c, ref_c)
    same_set = float(np.mean([set(a) == set(b) for a, b in zip(np.asarray(idx[rows]), ref_i)]))
    other = np.sort(np.random.default_rng(seed + 1).integers(0, n_rows, n_check))
    res['recompute'] = {'n_frames': n_check, 'rows': rows.tolist(), 'cos_min': float(cos.min()),
                        'same_atom_set_frac': same_set,
                        'baseline_random_rows_cos_median': float(_cos(torch.from_numpy(np.asarray(cs[other])), ref_c).median())}
    res['passed'] = bool(cs.shape[0] == n_rows and n_nan == 0 and float(cos.median()) > 0.99
                         and (res['codes_max_alignment_cos_median_min'] is None or res['codes_max_alignment_cos_median_min'] > 0.99))
    return res
