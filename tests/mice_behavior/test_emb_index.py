"""Tests for key-based lookup of cached embeddings (src/mice_behavior/emb_index.py) and the
loaders routed through it (src/mice_behavior/batch_data.py, src/mice_behavior/dataset.py).

The bug these guard against: the mice v1 caches (dinov2/dinov3 class_l-2 and patch_grid4)
were extracted before annotations.csv was rebuilt in a different row order, and loaders
indexed them by annotations row -> silently wrong frames.

(a) synthetic permuted cache -> CLS / patch-grid / concat loaders, FrameBatchData and
    MouseOPairDataset all return the frame each annotations row names
(b) missing row_keys.parquet -> hard error that says how to create it; duplicated keys -> error
(c) requested observation absent from the cache -> error naming it
(d) REAL data: 3 observations in the misaligned region; key-resolved loader output matches
    embeddings recomputed from the JPG frames (cos > 0.999) while the old positional lookup
    does not (cos < 0.99)
(e) REAL data: caches are complete -- class_l-2 resolves all 2,592,000 annotations rows
    (both encoders), the cls loader returns every frame of all 144 annotated observations
    (rd64 included), and patch_grid4 covers all 144 annotated observations (zero missing)

Runs standalone (`python tests/mice_behavior/test_emb_index.py`, prints measured numbers);
real-data tests need a GPU for speed (fall back to CPU) and the files under dataset/mice/v1.
Set EMB_INDEX_SKIP_REAL=1 to run only the synthetic tests.
"""
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'mice_behavior'))

from src.mice_behavior.batch_data import (  # noqa: E402
    FrameBatchData, load_cls_embeddings, load_patchgrid_concat_embeddings, load_patchgrid_embeddings,
)
from src.mice_behavior.dataset import MouseOPairDataset  # noqa: E402
from src.mice_behavior.emb_index import (  # noqa: E402
    EmbeddingIndexError, available_observations, resolve_rows, write_row_keys,
)

V1 = ROOT / 'dataset' / 'mice' / 'v1'
ANN_CSV = V1 / 'annotations.csv'
EMB = V1 / 'embeddings' / 'full'
D = 8


# ----------------------------------------------------------------------------- synthetic world
def _world(tmp: Path):
    """3 observations x 10 frames. annotations.csv in order A, B, C. Two caches:
    'cls'   : all 30 frames in a random permutation, CLS value row = [obs_code, frame_idx, 0...]
    'patch' : only A and C (B missing), shuffled, (n, 4, D) with the same encoding per patch."""
    rng = np.random.default_rng(0)
    obs = ['obsA', 'obsB', 'obsC']
    ann = pd.DataFrame({'observation_id': np.repeat(obs, 10), 'frame_idx': np.tile(np.arange(10), 3)})
    ds = tmp / 'ds'
    (ds / 'embeddings' / 'full' / 'enc' / 'cls').mkdir(parents=True)
    (ds / 'embeddings' / 'full' / 'enc' / 'patch').mkdir(parents=True)
    (ds / 'embeddings' / 'full' / 'enc2' / 'patch').mkdir(parents=True)
    ann_csv = ds / 'annotations.csv'
    ann.to_csv(ann_csv, index=False)
    # positives in every obs so all are "annotated"
    pl = pd.DataFrame({'observation_id': obs * 2, 'frame_idx': [3, 4, 5, 6, 7, 8],
                       'agent1': [0] * 6, 'agent2': [1] * 6, 'label': [1, 2, 1, 2, 1, 2]})
    pl_path = ds / 'pair_labels.parquet'
    pl.to_parquet(pl_path, index=False)

    def encode(keys):
        v = np.zeros((len(keys), D), dtype=np.float32)
        v[:, 0] = [obs.index(o) + 1 for o in keys['observation_id']]
        v[:, 1] = keys['frame_idx'].values
        v[:, 2] = 1.0
        return v

    perm = rng.permutation(len(ann))
    k_cls = ann.iloc[perm].reset_index(drop=True)
    cls_path = ds / 'embeddings' / 'full' / 'enc' / 'cls' / 'embeddings.npy'
    encode(k_cls).tofile(cls_path)  # headerless float32, like the real cache
    write_row_keys(cls_path.parent, k_cls['observation_id'], k_cls['frame_idx'])

    patch_paths = []
    for enc in ('enc', 'enc2'):
        sub = ann[ann['observation_id'] != 'obsB']
        k_p = sub.iloc[rng.permutation(len(sub))].reset_index(drop=True)
        pp = ds / 'embeddings' / 'full' / enc / 'patch' / 'embeddings.npy'
        np.repeat(encode(k_p)[:, None, :], 4, axis=1).astype(np.float16).tofile(pp)
        np.save(pp.parent / 'global_idx.npy', np.arange(len(k_p), dtype=np.int32))
        write_row_keys(pp.parent, k_p['observation_id'], k_p['frame_idx'])
        patch_paths.append(pp)
    return ann, ann_csv, pl_path, cls_path, patch_paths, obs


def _boundary(ann, obs_ids):
    b = {}
    for o in obs_ids:
        idx = np.flatnonzero(ann['observation_id'].values == o)
        b[o] = (int(idx[0]), int(idx[-1]) + 1)
    return b


def _check_block(ann, s, arr, obs):
    exp_code = np.array([obs.index(o) + 1 for o in ann['observation_id'].values[s:s + len(arr)]])
    exp_fr = ann['frame_idx'].values[s:s + len(arr)]
    a = np.asarray(arr, dtype=np.float32).reshape(len(arr), -1, arr.shape[-1])
    assert (a[:, :, 0] == exp_code[:, None]).all() and (a[:, :, 1] == exp_fr[:, None]).all()


def test_a_synthetic_permuted_cache():
    with tempfile.TemporaryDirectory() as t:
        ann, ann_csv, pl_path, cls_path, (pa, pb), obs = _world(Path(t))
        # CLS loader, explicit and inferred annotations_csv
        for kw in ({'annotations_csv': str(ann_csv)}, {}):
            out = load_cls_embeddings(str(cls_path), D, **kw)(_boundary(ann, obs))
            for s, arr in out.items():
                _check_block(ann, s, arr, obs)
        # the old positional read would be wrong on this cache
        raw = np.fromfile(cls_path, dtype=np.float32).reshape(-1, D)
        pos_wrong = int((raw[:, 1] != ann['frame_idx'].values).sum() + (raw[:, 0] != np.repeat([1, 2, 3], 10)).sum())
        assert pos_wrong > 0
        # patch-grid + concat loaders (the two patch caches have DIFFERENT row orders)
        b = _boundary(ann, ['obsA', 'obsC'])
        for s, arr in load_patchgrid_embeddings(str(pa), str(pa.parent / 'global_idx.npy'), 4, D,
                                                annotations_csv=str(ann_csv))(b).items():
            _check_block(ann, s, arr, obs)
        cat = load_patchgrid_concat_embeddings(str(pa), str(pb), None, 4, D, D, annotations_csv=str(ann_csv))(b)
        for s, arr in cat.items():  # each half is L2-normalised; ratio of dims 0/2 and 1/2 recovers the key
            a = arr.astype(np.float32)
            for half in (a[..., :D], a[..., D:]):
                code = np.rint(half[..., 0] / half[..., 2])
                fr = np.rint(half[..., 1] / half[..., 2])
                assert (code == np.array([obs.index(o) + 1 for o in ann['observation_id'].values[s:s + len(a)]])[:, None]).all()
                assert (fr == ann['frame_idx'].values[s:s + len(a)][:, None]).all()
        # end to end: FrameBatchData centre frame of every sample == its annotations row
        fb = FrameBatchData(str(ann_csv), str(pl_path), obs, 1, D,
                            load_cls_embeddings(str(cls_path), D, annotations_csv=str(ann_csv)))
        ctx, offs, _, mask = fb.get_batch(np.arange(len(fb)))
        centre = ctx[:, 1].numpy()
        assert (centre[:, 0] == np.array([obs.index(o) + 1 for o in ann['observation_id'].values[fb.gi]])).all()
        assert (centre[:, 1] == ann['frame_idx'].values[fb.gi]).all()
        # neighbours too (offset +1 where not padded)
        nb = ctx[:, 2].numpy()
        ok = ~mask[:, 2].numpy()
        assert (nb[ok, 1] == ann['frame_idx'].values[fb.gi[ok]] + 1).all()
        # MouseOPairDataset, preload and mmap modes
        for preload in (True, False):
            dsx = MouseOPairDataset(str(ann_csv), str(pl_path), str(cls_path), obs_ids=obs, context_k=1,
                                    emb_dim=D, preload=preload)
            for i in range(0, len(dsx), 7):
                gi = int(dsx.samples[i, 0])
                context, offsets, *_ = dsx[i]
                row = context.numpy()[list(offsets.numpy()).index(0)]
                assert row[0] == obs.index(ann['observation_id'].values[gi]) + 1 and row[1] == ann['frame_idx'].values[gi]
        print(f'    synthetic: positional read would mislabel {pos_wrong} of 60 key fields; '
              f'key-resolved loaders, FrameBatchData ({len(fb)} samples) and MouseOPairDataset all exact')


def test_b_missing_or_duplicate_sidecar_errors():
    with tempfile.TemporaryDirectory() as t:
        ann, ann_csv, pl_path, cls_path, _, obs = _world(Path(t))
        (cls_path.parent / 'row_keys.parquet').unlink()
        try:
            load_cls_embeddings(str(cls_path), D, annotations_csv=str(ann_csv))(_boundary(ann, obs))
        except EmbeddingIndexError as e:
            msg = str(e)
            assert 'row_keys.parquet not found' in msg and 'build_row_keys.py' in msg
        else:
            raise AssertionError('missing sidecar did not raise')
        try:
            write_row_keys(cls_path.parent, ['x', 'x'], [0, 0])
        except EmbeddingIndexError as e:
            assert 'duplicated' in str(e)
        else:
            raise AssertionError('duplicate keys were written')
        # a sidecar with duplicates written by other means is also refused at read time
        pd.DataFrame({'observation_id': ['obsA'] * 30, 'frame_idx': [0] * 30}).to_parquet(cls_path.parent / 'row_keys.parquet')
        try:
            resolve_rows(cls_path.parent, str(ann_csv))
        except EmbeddingIndexError as e:
            assert 'duplicated' in str(e)
        else:
            raise AssertionError('duplicate sidecar was accepted')
        print(f'    missing sidecar -> "{msg.splitlines()[0][:90]}..."')


def test_c_missing_observation_named():
    with tempfile.TemporaryDirectory() as t:
        ann, ann_csv, pl_path, cls_path, (pa, _), obs = _world(Path(t))
        try:
            load_patchgrid_embeddings(str(pa), None, 4, D, annotations_csv=str(ann_csv))(_boundary(ann, obs))
        except EmbeddingIndexError as e:
            msg = str(e)
            assert 'obsB (10 frames)' in msg and 'obsA' not in msg.split(':')[1]
        else:
            raise AssertionError('missing observation did not raise')
        assert available_observations(pa.parent, str(ann_csv)) == {'obsA', 'obsC'}
        # obs_boundary built from a different annotations table is caught
        try:
            load_cls_embeddings(str(cls_path), D, annotations_csv=str(ann_csv))({'obsA': (5, 15)})
        except EmbeddingIndexError as e:
            assert 'same annotations.csv' in str(e)
        else:
            raise AssertionError('mismatched obs_boundary accepted')
        print(f'    missing obs -> "{msg[:110]}"')


def test_f_extraction_writes_sidecar():
    """src/embedding/get_embeddings.py writes row_keys.parquet from the dataset's own key columns."""
    from datasets import Dataset
    from src.embedding.get_embeddings import _write_row_keys_sidecar
    from src.mice_behavior.emb_index import load_row_keys
    ds = Dataset.from_dict({'observation_id': ['b', 'a', 'b'], 'frame_idx': [1, 0, 0]})
    with tempfile.TemporaryDirectory() as t:
        _write_row_keys_sidecar(ds, Path(t), 3, verbose=False)
        k = load_row_keys(t)
        assert list(k['observation_id']) == ['b', 'a', 'b'] and list(k['frame_idx']) == [1, 0, 0]
        assert _write_row_keys_sidecar(Dataset.from_dict({'x': [1]}), Path(t) / 'none', 1) is None


# ----------------------------------------------------------------------------- real data
def _real_available():
    return os.environ.get('EMB_INDEX_SKIP_REAL') != '1' and ANN_CSV.exists() and \
        (EMB / 'dinov2' / 'class_l-2' / 'row_keys.parquet').exists()


def _cos(a, b):
    a, b = a.reshape(len(a), -1).astype(np.float64), b.reshape(len(b), -1).astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def test_d_real_fixed_vs_positional():
    if not _real_available():
        print('    SKIPPED (no real data)')
        return
    from build_row_keys import Recomputer
    ann = pd.read_csv(ANN_CSV, usecols=['observation_id', 'frame_idx', 'frame_path'])
    annotated = set(pd.read_parquet(V1 / 'pair_labels.parquet')['observation_id'])
    b_all = {}
    oid = ann['observation_id'].values
    starts = np.r_[0, np.flatnonzero(oid[1:] != oid[:-1]) + 1]
    for s, e in zip(starts, np.r_[starts[1:], len(oid)]):
        b_all[oid[s]] = (int(s), int(e))
    # 3 annotated observations in the misaligned region (annotations rows >= 612,000) present in both caches
    # only observations in the LEGACY part of the cache (rows appended later by
    # complete_emb_caches.py were never read by the old positional code)
    from src.mice_behavior.emb_index import load_row_keys
    pg_ok = set(load_row_keys(EMB / 'dinov2' / 'patch_grid4')['observation_id'].values[:774_000])
    # patch_grid4: only annotations rows < 774,000 can be looked up by the old code at all
    # (beyond that it raised KeyError), so the silently-wrong obs are those in [612,000, 774,000)
    def spread(hi):
        c = [o for o, (s, e) in sorted(b_all.items(), key=lambda kv: kv[1][0])
             if s >= 612_000 and e <= hi and o in annotated and o in pg_ok]
        return [c[0], c[len(c) // 2], c[-1]]
    rec = Recomputer('cuda')
    lines = []
    for enc, tok, kind in (('dinov2', 'class_l-2', 'cls'), ('dinov2', 'patch_grid4', 'patch')):
        pick = spread(len(oid) if kind == 'cls' else 774_000)
        b = {o: b_all[o] for o in pick}
        emb_path = EMB / enc / tok / 'embeddings.npy'
        if kind == 'cls':
            out = load_cls_embeddings(str(emb_path), 768, annotations_csv=str(ANN_CSV))(b)
            raw = np.memmap(emb_path, dtype='float32', mode='r', shape=(2_556_000, 768))
        else:
            out = load_patchgrid_embeddings(str(emb_path), str(emb_path.parent / 'global_idx.npy'), 16, 768,
                                            annotations_csv=str(ANN_CSV))(b)
            raw = np.memmap(emb_path, dtype='float16', mode='r', shape=(774_000, 16, 768))
            gidx = np.load(emb_path.parent / 'global_idx.npy')[:774_000]  # legacy entries = old frame-table rows
            row_of_global = {int(g): i for i, g in enumerate(gidx)}
        for o in pick:
            s, e = b[o]
            frames_local = np.array([17, (e - s) // 2, e - s - 3])
            ann_rows = s + frames_local
            fixed = out[s][frames_local].astype(np.float32)
            if kind == 'cls':  # OLD code: mmap[obs_s:obs_e] by annotations row
                old = np.asarray(raw[ann_rows], dtype=np.float32)
            else:              # OLD code: row_of_global[annotations row]
                old = np.asarray(raw[[row_of_global[int(r)] for r in ann_rows]], dtype=np.float32)
            fresh = rec(enc, kind, ann['frame_path'].values[ann_rows].tolist())
            c_fix, c_old = _cos(fixed, fresh), _cos(old, fresh)
            lines.append(f'    {enc}/{tok} {o} (ann rows {s:,}+{list(frames_local)}): fixed cos '
                         f'{c_fix.min():.5f}..{c_fix.max():.5f}, old positional {c_old.min():.4f}..{c_old.max():.4f}')
            assert (c_fix > 0.999).all(), (o, c_fix)
            assert (c_old < 0.99).all(), (o, c_old)
    print('\n'.join(lines))


def test_e_real_coverage_all_annotated():
    if not _real_available():
        print('    SKIPPED (no real data)')
        return
    ann = pd.read_csv(ANN_CSV, usecols=['observation_id', 'frame_idx'])
    annotated = sorted(set(pd.read_parquet(V1 / 'pair_labels.parquet')['observation_id']))
    assert len(annotated) == 144
    oid = ann['observation_id'].values
    b = {}
    for o in annotated:
        idx = np.flatnonzero(oid == o)
        b[o] = (int(idx[0]), int(idx[-1]) + 1)
    # class_l-2: every annotations.csv frame (all 432 observations) is cached, for both encoders
    for enc in ('dinov2', 'dinov3'):
        d = EMB / enc / 'class_l-2'
        resolved = resolve_rows(d, str(ANN_CSV))
        n_miss = int((resolved < 0).sum())
        print(f'    class_l-2 ({enc}): {len(resolved):,} annotations rows, {n_miss} unresolved, '
              f'{len(set(oid) - available_observations(d, str(ANN_CSV)))} observations missing')
        assert n_miss == 0 and len(resolved) == 2_592_000
    emb_path = EMB / 'dinov2' / 'class_l-2' / 'embeddings.npy'
    t0 = time.time()
    out = load_cls_embeddings(str(emb_path), 768, annotations_csv=str(ANN_CSV))(b)  # all 144, incl. rd64
    counts = pd.Series({o: len(out[s]) for o, (s, e) in b.items()})
    assert (counts == np.array([e - s for s, e in b.values()])).all()
    assert all(np.isfinite(a).all() and (np.abs(a).sum(1) > 0).all() for a in out.values())
    print(f'    class_l-2 (dinov2) loader: all {len(b)} annotated obs loaded ({counts.sum():,} frames; '
          f'per-obs count min {counts.min()} max {counts.max()})  [{time.time() - t0:.0f}s]')
    for enc in ('dinov2', 'dinov3'):
        pg_ok = available_observations(EMB / enc / 'patch_grid4', str(ANN_CSV))
        missing_pg = sorted(set(annotated) - pg_ok)
        print(f'    patch_grid4 ({enc}): {len(set(annotated) & pg_ok)} of {len(annotated)} annotated obs covered, '
              f'{len(missing_pg)} missing: {missing_pg}')
        assert not missing_pg


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    if len(sys.argv) > 1:  # optional name filter, e.g. `python ... test_d`
        tests = [t for t in tests if any(a in t.__name__ for a in sys.argv[1:])]
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f'  PASS  {t.__name__}  ({time.time() - t0:.1f}s)')
        except Exception:
            failed += 1
            print(f'  FAIL  {t.__name__}')
            traceback.print_exc()
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
