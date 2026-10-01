"""Tests of the spatial frame features (src/eci/spatial_encode.py) against brute-force numpy loops.

  (a) pair_strength: random sparse patch codes (5-40% of patches foreground, 1-6 active latents each among L = 128)
      equal a loop over every ordered pair of 8-neighbouring patches (p != q) and every latent pair a != b,
      s_{min(a,b), max(a,b)} = max min(z_a(p), z_b(q)), to float precision; truncation count is 0 there.
  (b) zone_max: equals a per-zone loop; zone_maps('ants') is the 3 x 3 grid, 'mice' uses the odor table radius.
  (c) components / blob_stats: labels equal scipy.ndimage.label with 8-connectivity (as partitions); n_blobs,
      mean pairwise centroid distance, n_components, largest size equal a loop.

Runs standalone (`python tests/eci/test_spatial.py`) or under pytest.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eci.spatial_encode import GRID, P, blob_stats, components, pair_strength, zone_maps, zone_max  # noqa: E402

NB8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def random_frames(B=6, L=128, seed=0):
    rng = np.random.default_rng(seed)
    Z = np.zeros((B, P, L), np.float32)
    masks = np.zeros((B, P), bool)
    for b in range(B):
        # blobs: a few random rectangles + speckle
        m = np.zeros((GRID, GRID), bool)
        for _ in range(rng.integers(1, 6)):
            y, x = rng.integers(0, GRID, 2)
            m[y:y + rng.integers(1, 6), x:x + rng.integers(1, 6)] = True
        m |= rng.random((GRID, GRID)) < 0.03
        masks[b] = m.ravel()
        for p in np.flatnonzero(masks[b]):
            idx = rng.choice(L, rng.integers(1, 7), replace=False)
            Z[b, p, idx] = rng.random(len(idx)).astype(np.float32) + 0.01
    return Z, masks


def pair_loop(Zb, L):
    s = np.zeros((L, L), np.float32)
    for p in range(P):
        y, x = divmod(p, GRID)
        ap = np.flatnonzero(Zb[p] > 0)
        if not len(ap):
            continue
        for dy, dx in NB8:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < GRID and 0 <= xx < GRID):
                continue
            q = yy * GRID + xx
            for b in np.flatnonzero(Zb[q] > 0):
                for a in ap:
                    if a == b:
                        continue
                    lo, hi = min(a, b), max(a, b)
                    s[lo, hi] = max(s[lo, hi], min(Zb[p, a], Zb[q, b]))
    iu = np.triu_indices(L, 1)
    return s[iu]


def test_pairs():
    L = 128
    Z, _ = random_frames(L=L)
    got, trunc = pair_strength(torch.from_numpy(Z))
    assert trunc == 0
    for b in range(len(Z)):
        ref = pair_loop(Z[b], L)
        assert np.allclose(got[b].numpy(), ref, atol=1e-6), np.abs(got[b].numpy() - ref).max()
        assert (ref > 0).sum() > 0
    print('pairs ok:', [int((got[b] > 0).sum()) for b in range(len(Z))], 'nonzero pairs per frame')


def test_zones():
    Z, masks = random_frames(L=16, seed=1)
    zm, names = zone_maps('ants', ['a', 'b'])
    assert len(names) == 9 and zm.shape == (2, P) and set(np.unique(zm)) == set(range(9))
    g = zm[0].reshape(GRID, GRID)
    assert g[0, 0] == 0 and g[0, 31] == 2 and g[31, 0] == 6 and g[15, 15] == 4 and g[10, 10] == 0 and g[11, 11] == 4
    zones = torch.from_numpy(np.tile(zm[0], (len(Z), 1)).astype(np.int64))
    got = zone_max(torch.from_numpy(Z), zones, 9).numpy().reshape(len(Z), 9, 16)
    for b in range(len(Z)):
        for z in range(9):
            sel = zm[0] == z
            assert np.allclose(got[b, z], Z[b, sel].max(0))
    odor = ROOT / 'dataset/mice/v1/eci/odor_corner.csv'
    if odor.exists():
        import pandas as pd
        t = pd.read_csv(odor)
        zm2, names2 = zone_maps('mice', t['observation_id'].values[:5], odor)
        assert names2 == ('near_odor', 'rest')
        for k in range(5):
            r = t.iloc[k]
            p = int(r['corner_y_patch']) * GRID + int(r['corner_x_patch'])
            p = min(max(p, 0), P - 1)
            assert zm2[k].sum() < P  # some near patches
        print('mice near-odor patches:', (zm2 == 0).sum(1))
    print('zones ok')


def test_blobs():
    from scipy import ndimage
    _, masks = random_frames(B=12, L=8, seed=2)
    lab = components(torch.from_numpy(masks)).numpy()
    st = blob_stats(torch.from_numpy(masks)).numpy()
    for b in range(len(masks)):
        ref, n = ndimage.label(masks[b].reshape(GRID, GRID), structure=np.ones((3, 3)))
        ref = ref.ravel()
        # same partition
        pairs = set(zip(ref[masks[b]], lab[b][masks[b]]))
        assert len(pairs) == n == len(set(lab[b][masks[b]]))
        sizes = np.bincount(ref)[1:]
        big = np.flatnonzero(sizes >= 2) + 1
        cents = [np.array(np.unravel_index(np.flatnonzero(ref == k), (GRID, GRID))).mean(1) for k in big]
        d = [np.hypot(*(cents[i] - cents[j])) for i in range(len(cents)) for j in range(len(cents)) if i != j]
        assert st[b, 0] == len(big) and st[b, 2] == n and st[b, 3] == sizes.max()
        assert np.isclose(st[b, 1], np.mean(d) if d else 0.0, atol=1e-4), (st[b, 1], np.mean(d) if d else 0)
    empty = blob_stats(torch.zeros(1, P, dtype=torch.bool)).numpy()
    assert (empty == 0).all()
    print('blobs ok:', st[:, 0].tolist())


if __name__ == '__main__':
    test_pairs()
    test_zones()
    test_blobs()
    print('all ok')
