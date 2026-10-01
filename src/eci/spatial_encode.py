"""
Spatial frame features that keep per-patch information (ECI): one GPU pass per foreground SAE writes three new
per-frame feature sets, computed from the same tokens, foreground mask and SAE patch codes as the stored codes
(src/eci/fg_encode.py _Runner: DINOv2 at 448, the SAE's foreground rule, threshold inference).

Patch grid: 32 x 32 patches (16 x 16 px of the 512 px frame each), patch index p = row * 32 + col. Patch codes
z(p) (1024 latents, >= 0) exist on foreground patches only; off-mask patches count as 0. Only the first
PREFIX = 128 Matryoshka latents enter the pair and zone features.

1. Pairwise co-activation (`pair_strength`). For an UNORDERED pair of distinct latents a < b (both < 128):
       s_ab(frame) = max over 8-neighbouring patch pairs (p, q), p != q, of min(z_a(p), z_b(q)) [and min(z_b(p), z_a(q))]
   i.e. the strength at which a fires on one patch and b fires on a patch touching it (side or corner). 0 = never.
   The same patch (p = q) is not a co-activation: that is ordinary code co-occurrence, which max pooling already
   sees. a = b (one latent on two touching patches = concept extent) is not included. The frame indicator is
   s_ab > 0. Columns: all 8128 pairs a < b in (a, b) order (written to shards; the merge keeps the pairs with
   s_ab > 0 in >= 1% of all frames).
   Implementation: per patch the PAIR_K largest of its first-128 codes (more than any patch has active in practice;
   the count of patches with more active latents than PAIR_K is recorded per shard, `pair_truncated_patches`).
2. Zone pooling (`zone_max`). zones (P,) int per video (-1 = no zone): per frame, zone and latent < 128, the max of
   z over the frame's foreground patches inside the zone (0 when none). Columns zone-major: zone * 128 + latent.
   Per frame also the foreground patch count per zone (zone_nfg).
     mice  2 zones per video from dataset/mice/v1/eci/odor_corner.csv (scripts/eci/mice_odor_corner.py):
           0 = near odor (patch centre within 0.5 x arena side of the arena corner on the odor-bag side), 1 = rest
     ants  3 x 3 grid of the frame (patch rows / cols 0-10, 11-21, 22-31), zone = 3 * grid row + grid col
3. Blob measures (`blob_stats`), from the foreground mask alone: connected components of the mask on the 32 x 32
   grid, 8-connectivity (as the mask's own 3 x 3 dilation / isolation rules); a blob = a component of >= MIN_BLOB = 2
   patches (single patches are mostly mask noise). Per frame: [n_blobs, mean pairwise distance between blob
   centroids (patch units; 0 when < 2 blobs), n_components (any size), largest component (patches)].

Alignment check: every batch also max-pools the SAE codes and compares them with the stored codes_max rows
(cosine per frame); a batch median below 0.99 stops the run (misaligned rows), as src/eci/somp_encode.py.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

GRID = 32
P = GRID * GRID
PREFIX = 128
PAIR_K = 24
MIN_BLOB = 2
# half of the 8-neighbourhood: every unordered pair of touching patches appears once
OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))
TRIU = np.triu_indices(PREFIX, 1)  # (a, b), a < b: 8128 pairs, column order of the pair features
TRIU_KEY = torch.from_numpy(TRIU[0] * PREFIX + TRIU[1])


def _shift_slices(dy, dx, g=GRID):
    ya, yb = slice(0, g - dy), slice(dy, g)
    xa, xb = (slice(0, g - dx), slice(dx, g)) if dx >= 0 else (slice(-dx, g), slice(0, g + dx))
    return ya, xa, yb, xb


@torch.no_grad()
def pair_strength(Z, k=PAIR_K):
    """Z (B, P, L) float (0 off-mask) -> (B, L * (L - 1) / 2) float32 pair strengths s_ab (a < b, TRIU order) and
    the number of patches with more than k active latents (truncated)."""
    B, _, L = Z.shape
    k = min(k, L)
    trunc = int(((Z > 0).sum(2) > k).sum())
    v, i = Z.float().topk(k, dim=2)
    v, i = v.view(B, GRID, GRID, k), i.view(B, GRID, GRID, k)
    out = torch.zeros(B, L * L, device=Z.device)
    for dy, dx in OFFSETS:
        ya, xa, yb, xb = _shift_slices(dy, dx)
        va, vb = v[:, ya, xa][..., :, None], v[:, yb, xb][..., None, :]
        ia, ib = i[:, ya, xa][..., :, None], i[:, yb, xb][..., None, :]
        val = torch.minimum(va, vb)
        key = torch.minimum(ia, ib) * L + torch.maximum(ia, ib)
        out.scatter_reduce_(1, key.reshape(B, -1), val.reshape(B, -1), 'amax', include_self=True)
    tk = (torch.from_numpy(np.triu_indices(L, 1)[0] * L + np.triu_indices(L, 1)[1]) if L != PREFIX else TRIU_KEY)
    return out[:, tk.to(Z.device)], trunc


@torch.no_grad()
def zone_max(Z, zones, n_zones):
    """Z (B, P, L), zones (B, P) long (-1 none) -> (B, n_zones * L) max per zone (zone-major), (B, n_zones) counts.
    Z must be 0 off the foreground mask; zone counts count foreground patches (fg (B, P) bool via Z's caller)."""
    B, _, L = Z.shape
    out = torch.zeros(B, n_zones, L, device=Z.device)
    for z in range(n_zones):
        m = (zones == z)
        out[:, z] = Z.masked_fill(~m[..., None], 0).amax(1)
    return out.reshape(B, n_zones * L)


@torch.no_grad()
def components(mask):
    """(B, P) bool -> (B, P) long component labels (0 = background, else the largest patch index + 1 in the
    component), 8-connectivity on the 32 x 32 grid (max-label propagation to convergence)."""
    B = mask.shape[0]
    m = mask.view(B, 1, GRID, GRID).float()
    lab = torch.arange(1, P + 1, device=mask.device, dtype=torch.float32).view(1, 1, GRID, GRID) * m
    while True:
        new = F.max_pool2d(lab, 3, 1, 1) * m
        if torch.equal(new, lab):
            break
        lab = new
    return lab.view(B, P).long()


@torch.no_grad()
def blob_stats(mask, min_size=MIN_BLOB):
    """(B, P) bool -> (B, 4) float32 [n_blobs (>= min_size patches), mean pairwise centroid distance between those
    blobs (patch units, 0 if < 2), n_components (any size), largest component size]."""
    B = mask.shape[0]
    lab = components(mask)
    ones = torch.ones_like(lab, dtype=torch.float32)
    yy = (torch.arange(P, device=mask.device) // GRID).float().expand(B, P)
    xx = (torch.arange(P, device=mask.device) % GRID).float().expand(B, P)
    cnt = torch.zeros(B, P + 1, device=mask.device).scatter_add_(1, lab, ones)[:, 1:]
    sy = torch.zeros(B, P + 1, device=mask.device).scatter_add_(1, lab, yy)[:, 1:]
    sx = torch.zeros(B, P + 1, device=mask.device).scatter_add_(1, lab, xx)[:, 1:]
    big = cnt >= min_size
    nb = big.sum(1)
    kc = max(int(nb.max()), 1)
    order = torch.argsort(big.to(torch.int8), dim=1, descending=True, stable=True)[:, :kc]
    c = torch.stack([sy.gather(1, order), sx.gather(1, order)], 2) / cnt.gather(1, order).clamp_min(1)[..., None]
    valid = torch.arange(kc, device=mask.device)[None] < nb[:, None]
    d = torch.cdist(c, c)
    pm = valid[:, :, None] & valid[:, None, :] & ~torch.eye(kc, dtype=torch.bool, device=mask.device)[None]
    npair = pm.sum((1, 2)).float()
    mean_d = torch.where(npair > 0, (d * pm).sum((1, 2)) / npair.clamp_min(1), torch.zeros_like(npair))
    return torch.stack([nb.float(), mean_d, (cnt > 0).sum(1).float(), cnt.max(1).values], 1)


BLOB_COLUMNS = ('n_blobs', 'mean_blob_distance', 'n_components', 'largest_component')


def zone_maps(domain, obs_ids, odor_csv=None):
    """-> (zones (n_videos, P) int8 aligned with obs_ids, zone names). mice: near odor / rest per video from the
    odor-corner table; ants: 3 x 3 grid."""
    if domain == 'mice':
        t = pd.read_csv(odor_csv).set_index('observation_id')
        c = np.arange(GRID) + 0.5
        yy, xx = np.meshgrid(c, c, indexing='ij')
        out = np.zeros((len(obs_ids), P), np.int8)
        for k, o in enumerate(obs_ids):
            r = t.loc[o]
            near = np.hypot(yy - r['corner_y_patch'], xx - r['corner_x_patch']).ravel() <= r['near_radius_patch']
            out[k] = np.where(near, 0, 1)
        return out, ('near_odor', 'rest')
    edges = np.array([0, 11, 22, GRID])
    g = np.searchsorted(edges, np.arange(GRID), side='right') - 1
    z = (3 * g[:, None] + g[None, :]).ravel().astype(np.int8)
    names = tuple(f'{r}{c}' for r in ('top', 'mid', 'bottom') for c in ('_left', '_centre', '_right'))
    return np.tile(z, (len(obs_ids), 1)), names


class SpatialEncoder:
    """Tokens, mask and SAE patch codes with the SAE's own pipeline (fg_encode._Runner), then the three feature sets."""

    def __init__(self, sae_path, frame_paths, dataset_dir, bg_dir, ann_path, domain, odor_csv=None, device='cuda'):
        from src.eci.fg_encode import _Runner
        self.device = torch.device(device)
        self.run = _Runner([sae_path], bg_dir, ann_path, device, frame_paths, dataset_dir)
        if self.run.deltas != [0]:
            raise ValueError('static-token SAEs only (motion_delta 0)')
        self.sae, self.norm = self.run.saes[0]
        self.m = self.sae.n_latents
        self.frame_paths, self.dataset_dir = frame_paths, Path(dataset_dir)
        zm, self.zone_names = zone_maps(domain, self.run.bgs.ids, odor_csv)
        self.zones = torch.from_numpy(zm.astype(np.int64)).to(self.device)
        self.n_zones = len(self.zone_names)

    def loader(self, rows, batch_size, num_workers):
        paths = [str(self.dataset_dir / self.frame_paths[r]) for r in rows]
        return self.run.loader(paths, np.asarray(rows), batch_size, num_workers)

    @torch.no_grad()
    def batch(self, item):
        from src.eci.foreground import encode_batch
        pix, grey, rows = item[0], item[1], item[2].numpy()
        tok = encode_batch(self.run.model, pix, self.device)
        mask, _ = self.run.bgs.mask(tok, grey.to(self.device, non_blocking=True), rows)
        tok = self.run.fg.sae_tokens(tok, item[3] if len(item) > 3 else None)
        B = tok.shape[0]
        fi, pi = torch.nonzero(mask, as_tuple=True)
        z = self.sae.encode(self.norm(tok[fi, pi]), mode='threshold')
        cmax = torch.zeros(B, self.m, device=self.device).index_reduce_(0, fi, z, 'amax', include_self=True)
        Z = torch.zeros(B, P, PREFIX, device=self.device)
        Z[fi, pi] = z[:, :PREFIX]
        pairs, trunc = pair_strength(Z)
        zones = self.zones[torch.from_numpy(self.run.bgs.obs_index(rows)).to(self.device)]  # (B, P)
        zonef = zone_max(Z, zones, self.n_zones)
        znfg = torch.stack([(mask & (zones == k)).sum(1) for k in range(self.n_zones)], 1)
        return {'pairs': pairs, 'zones': zonef, 'zone_nfg': znfg, 'blobs': blob_stats(mask), 'n_fg': mask.sum(1),
                'codes_max': cmax, 'truncated': trunc}


OUTPUTS = {'pairs': np.float16, 'zones': np.float16, 'zone_nfg': np.int16, 'blobs': np.float32, 'n_fg': np.int16}


def _cos(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=1, eps=1e-8)


def encode_spatial_shard(enc, lo, hi, shard_dir, ref_codes_max, batch_size=128, num_workers=16):
    """Rows [lo, hi) -> shard_dir (memmaps + shard.json + DONE); a finished shard is skipped."""
    shard_dir = Path(shard_dir)
    if (shard_dir / 'DONE').exists():
        print(f'[SKIP] {shard_dir} done')
        return shard_dir
    tmp = Path(str(shard_dir) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    N = hi - lo
    widths = {'pairs': len(TRIU[0]), 'zones': enc.n_zones * PREFIX, 'zone_nfg': enc.n_zones, 'blobs': len(BLOB_COLUMNS)}
    mm = {k: np.lib.format.open_memmap(tmp / f'{k}.npy', 'w+', dt, (N, widths[k]) if k in widths else (N,))
          for k, dt in OUTPUTS.items()}
    ref = np.load(ref_codes_max, mmap_mode='r')
    rows = np.arange(lo, hi)
    cos_all, trunc, t0, cur = [], 0, time.time(), 0
    for b, item in enumerate(enc.loader(rows, batch_size, num_workers)):
        assert int(item[2][0]) == lo + cur
        Bn = len(item[2])
        out = enc.batch(item)
        cos_all.append(_cos(out['codes_max'].half(),
                            torch.from_numpy(np.asarray(ref[lo + cur:lo + cur + Bn])).to(enc.device)).cpu())
        for k, dt in OUTPUTS.items():
            mm[k][cur:cur + Bn] = out[k].cpu().numpy().astype(dt)
        trunc += out['truncated']
        cur += Bn
        if b % 100 == 0:
            c = torch.cat(cos_all)
            print(f'  batch {b:5d}  {cur:7d}/{N}  {cur / (time.time() - t0):6.1f} frames/s  codes_max cos min '
                  f'{float(c.min()):.4f} median {float(c.median()):.5f}  truncated patches {trunc}', flush=True)
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
        'lo': lo, 'hi': hi, 'elapsed_s': round(elapsed, 1), 'pair_truncated_patches': trunc,
        'codes_max_alignment_cos': {'min': float(c.min()), 'p1': float(np.percentile(c, 1)), 'median': float(np.median(c)),
                                    'frac_below_0.99': float((c < 0.99).mean())}}))
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    tmp.rename(shard_dir)
    (shard_dir / 'DONE').touch()
    print(f'Done rows [{lo}, {hi}) in {elapsed:.0f}s ({N / elapsed:.1f} frames/s) -> {shard_dir}')
    return shard_dir


def _shards(src, ranges):
    d = [Path(src) / 'shards' / f'shard_{i:02d}' for i in range(len(ranges))]
    missing = [str(x) for x in d if not (x / 'DONE').exists()]
    if missing:
        raise RuntimeError(f'shards not finished: {missing}')
    return d


def _write(dst_path, shards, name, ranges, n_rows, cols=None):
    first = np.load(shards[0] / f'{name}.npy', mmap_mode='r')
    shape = (n_rows,) + ((len(cols),) if cols is not None else first.shape[1:])
    out = np.lib.format.open_memmap(str(dst_path) + '.tmp.npy', 'w+', first.dtype, shape)
    for s, (lo, hi) in zip(shards, ranges):
        a = np.load(s / f'{name}.npy', mmap_mode='r')
        assert a.shape[0] == hi - lo, (name, s, a.shape)
        for i in range(0, hi - lo, 32768):
            x = np.asarray(a[i:i + 32768])
            out[lo + i:lo + i + len(x)] = x[:, cols] if cols is not None else x
    out.flush()
    del out
    Path(str(dst_path) + '.tmp.npy').rename(dst_path)


def merge_spatial(src, ranges, n_rows, sae, codes_root, zone_names, min_frac=0.01):
    """Shards in <src>/shards -> <codes_root>/<sae>_pairs/, _zones/, _blobs/ (each with n_fg.npy, features.csv,
    config.json, DONE) and <src>/pair_frequency.csv (frame fraction with s_ab > 0 for all 8128 pairs)."""
    shards = _shards(src, ranges)
    codes_root = Path(codes_root)
    n_pos = np.zeros(len(TRIU[0]), np.int64)
    for s in shards:
        a = np.load(s / 'pairs.npy', mmap_mode='r')
        for i in range(0, a.shape[0], 32768):
            n_pos += (np.asarray(a[i:i + 32768]) > 0).sum(0)
    freq = n_pos / n_rows
    pf = pd.DataFrame({'column_all': np.arange(len(freq)), 'a': TRIU[0], 'b': TRIU[1], 'frame_frac': freq})
    pf.to_csv(Path(src) / 'pair_frequency.csv', index=False)
    keep = np.flatnonzero(freq >= min_frac)
    print(f'pairs: {len(keep)} of {len(freq)} occur in >= {min_frac:.0%} of frames', flush=True)
    sets = {
        'pairs': ('codes_pairs.npy', 'pairs', keep,
                  pf.iloc[keep].reset_index(drop=True).assign(column=np.arange(len(keep)))[['column', 'a', 'b', 'frame_frac', 'column_all']]),
        'zones': ('codes_zones.npy', 'zones', None,
                  pd.DataFrame({'column': np.arange(len(zone_names) * PREFIX),
                                'zone': np.repeat(np.arange(len(zone_names)), PREFIX),
                                'zone_name': np.repeat(zone_names, PREFIX), 'latent': np.tile(np.arange(PREFIX), len(zone_names))})),
        'blobs': ('blobs.npy', 'blobs', None, pd.DataFrame({'column': np.arange(len(BLOB_COLUMNS)), 'name': BLOB_COLUMNS}))}
    for tag, (fname, name, cols, feat) in sets.items():
        dst = codes_root / f'{sae}_{tag}'
        if (dst / 'DONE').exists():
            print(f'[SKIP] {dst} done')
            continue
        dst.mkdir(parents=True, exist_ok=True)
        _write(dst / fname, shards, name, ranges, n_rows, cols)
        _write(dst / 'n_fg.npy', shards, 'n_fg', ranges, n_rows)
        if tag == 'zones':
            _write(dst / 'zone_nfg.npy', shards, 'zone_nfg', ranges, n_rows)
        feat.to_csv(dst / 'features.csv', index=False)
        (dst / 'config.json').write_text(json.dumps({'source': str(src), 'set': tag, 'file': fname,
                                                     'see': 'src/eci/spatial_encode.py docstring',
                                                     'source_config': json.loads((Path(src) / 'config.json').read_text())},
                                                    indent=1))
        (dst / 'DONE').touch()
        print(f'  wrote {dst}', flush=True)
    (Path(src) / 'DONE').touch()


def verify_spatial(src, enc, sae, codes_root, n_rows, ranges, n_check=64, seed=0):
    """Per-shard alignment summary + n_check random rows recomputed from scratch and compared with the final arrays."""
    codes_root = Path(codes_root)
    al = [json.loads((Path(src) / 'shards' / f'shard_{i:02d}' / 'shard.json').read_text()) for i in range(len(ranges))]
    res = {'codes_max_alignment_cos_min': min(a['codes_max_alignment_cos']['min'] for a in al),
           'codes_max_alignment_cos_median_min': min(a['codes_max_alignment_cos']['median'] for a in al),
           'codes_max_alignment_frac_below_0.99_max': max(a['codes_max_alignment_cos']['frac_below_0.99'] for a in al),
           'pair_truncated_patches': sum(a['pair_truncated_patches'] for a in al)}
    feat = pd.read_csv(codes_root / f'{sae}_pairs' / 'features.csv')
    st = {'pairs': np.load(codes_root / f'{sae}_pairs' / 'codes_pairs.npy', mmap_mode='r'),
          'zones': np.load(codes_root / f'{sae}_zones' / 'codes_zones.npy', mmap_mode='r'),
          'blobs': np.load(codes_root / f'{sae}_blobs' / 'blobs.npy', mmap_mode='r'),
          'n_fg': np.load(codes_root / f'{sae}_blobs' / 'n_fg.npy', mmap_mode='r')}
    rows = np.sort(np.random.default_rng(seed).choice(n_rows, n_check, replace=False))
    got = {k: [] for k in st}
    for item in enc.loader(rows, 16, 8):
        o = enc.batch(item)
        o['pairs'] = o['pairs'][:, torch.from_numpy(feat['column_all'].values).to(enc.device)]
        for k in got:
            got[k].append(o[k].float().cpu())
    rec = {}
    for k in got:
        g = torch.cat(got[k]).numpy().astype(np.float32)
        s = np.asarray(st[k][rows], dtype=np.float32)
        g = g if g.ndim == 2 else g[:, None]
        s = s if s.ndim == 2 else s[:, None]
        rec[k] = {'max_abs_diff': float(np.abs(g - s).max()),
                  'frames_identical_frac': float((np.abs(g - s).max(1) <= (1e-2 if k in ('pairs', 'zones') else 1e-4)).mean())}
    res['recompute'] = {'n_frames': n_check, 'rows': rows.tolist(), **rec}
    # border-patch mask flips (GPU numerics) can change a frame: most frames must reproduce exactly
    res['passed'] = bool(res['codes_max_alignment_cos_median_min'] > 0.99
                         and all(v['frames_identical_frac'] >= 0.9 for v in rec.values()))
    return res
