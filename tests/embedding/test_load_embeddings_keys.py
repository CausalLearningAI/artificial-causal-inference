"""Tests for the frame-key safety check in src/embedding/get_embeddings.py:load_embeddings_from_disk.

The loader used by PPCI (src/ppci/dataset.py:PPCIDataset.from_disk) reads embeddings.pt/.npy
positionally. When a cache carries a row_keys.parquet sidecar (row i = (observation_id, frame_idx)
of embedding row i), the loader now checks those keys against the frame table the caller aligns to
and reorders or fails instead of silently returning the wrong frames.

(a) no sidecar            -> output identical to the stored tensor (old behaviour), one warning per cache
(b) sidecar in table order -> output identical to the stored tensor
(c) sidecar permuted       -> rows reordered to the table order (default annotations.csv, an HF
                              Dataset, a DataFrame, add_embeddings_from_disk)
(d) sidecar missing rows   -> EmbeddingIndexError naming the missing observation; sidecar with the
                              wrong row count -> error
(e) REAL data (mice v1 dinov2 class_l-2): the old positional read is wrong at annotations rows
    >= 612,000 (cos < 0.99 vs embeddings recomputed from the JPG frames) and the new loader is
    right (cos > 0.999). Needs dataset/mice/v1 and ~16 GB RAM; GPU for speed.
    Set EMB_KEYS_SKIP_REAL=1 to run only the synthetic tests.

Runs standalone: `python tests/embedding/test_load_embeddings_keys.py` (prints measured numbers).
"""
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'mice_behavior'))

from datasets import Dataset  # noqa: E402

import src.embedding.get_embeddings as ge  # noqa: E402
from src.embedding.get_embeddings import add_embeddings_from_disk, load_embeddings_from_disk  # noqa: E402
from src.mice_behavior.emb_index import EmbeddingIndexError, write_row_keys  # noqa: E402

D = 8
OBS = ['obsA', 'obsB', 'obsC']


def _encode(keys: pd.DataFrame) -> torch.Tensor:
    """Row = [obs_code, frame_idx, 1, noise...] so each row names its own frame."""
    v = np.random.default_rng(1).normal(size=(len(keys), D)).astype(np.float32)
    v[:, 0] = [OBS.index(o) + 1 for o in keys['observation_id']]
    v[:, 1] = keys['frame_idx'].values
    v[:, 2] = 1.0
    return torch.from_numpy(v)


def _world(tmp: Path, cache_keys: str, with_sidecar: bool):
    """dataset_root=tmp; subject 's', version 'v'; annotations.csv = A, B, C x 10 frames.
    cache_keys: 'same' (table order), 'perm' (random permutation), 'noB' (obsB absent, shuffled)."""
    ann = pd.DataFrame({'observation_id': np.repeat(OBS, 10), 'frame_idx': np.tile(np.arange(10), 3)})
    vdir = tmp / 's' / 'v'
    emb_dir = vdir / 'embeddings' / 'full' / 'enc' / 'tok'
    emb_dir.mkdir(parents=True)
    ann.to_csv(vdir / 'annotations.csv', index=False)
    rng = np.random.default_rng(0)
    if cache_keys == 'same':
        keys = ann
    elif cache_keys == 'perm':
        keys = ann.iloc[rng.permutation(len(ann))].reset_index(drop=True)
    else:
        sub = ann[ann['observation_id'] != 'obsB']
        keys = sub.iloc[rng.permutation(len(sub))].reset_index(drop=True)
    emb = _encode(keys)
    torch.save(emb, emb_dir / 'embeddings.pt')
    if with_sidecar:
        write_row_keys(emb_dir, keys['observation_id'], keys['frame_idx'])
    return ann, emb, emb_dir


def _names_rows(out: torch.Tensor, table: pd.DataFrame) -> bool:
    o = out.numpy()
    return (o[:, 0] == np.array([OBS.index(x) + 1 for x in table['observation_id']])).all() and \
        (o[:, 1] == table['frame_idx'].values).all()


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _load(tmp, **kw):
    return load_embeddings_from_disk('s', 'v', 'enc', 'tok', dataset_root=str(tmp), **kw)


def test_a_no_sidecar_unchanged():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        ann, emb, emb_dir = _world(t, 'perm', with_sidecar=False)
        cap = _Capture()
        ge.logger.addHandler(cap)
        try:
            outs = [_load(t), _load(t), _load(t, align_to=ann), _load(t, align_to=Dataset.from_pandas(ann))]
        finally:
            ge.logger.removeHandler(cap)
        for out in outs:  # stored order, bit-for-bit, whatever align_to says (no keys to check against)
            assert out.dtype == emb.dtype and torch.equal(out, emb)
        assert not _names_rows(outs[0], ann)  # (this cache really is permuted)
        warns = [m for m in cap.msgs if 'row_keys.parquet' in m]
        assert len(warns) == 1 and str(emb_dir) in warns[0], cap.msgs
        # .npy-only cache (no .pt) and HF-dataset-only cache also pass through unchanged
        (emb_dir / 'embeddings.pt').unlink()
        np.save(emb_dir / 'embeddings.npy', emb.numpy())
        assert torch.equal(_load(t), emb)
        print(f'    no sidecar: 4 calls returned the stored tensor exactly; {len(warns)} warning: '
              f'"{warns[0][:100]}..."')


def test_b_sidecar_in_order_unchanged():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        ann, emb, _ = _world(t, 'same', with_sidecar=True)
        for kw in ({}, {'align_to': ann}, {'align_to': Dataset.from_pandas(ann)},
                   {'align_to': str(t / 's' / 'v' / 'annotations.csv')}):
            out = _load(t, **kw)
            assert out.dtype == emb.dtype and torch.equal(out, emb), kw
            assert _names_rows(out, ann)
        print('    sidecar in table order: returned unchanged for default / DataFrame / Dataset / CSV align_to')


def test_c_sidecar_permuted_reordered():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        ann, emb, emb_dir = _world(t, 'perm', with_sidecar=True)
        out = _load(t)  # default: annotations.csv
        assert out.shape == emb.shape and _names_rows(out, ann)
        assert not torch.equal(out, emb)
        # a caller table in yet another order (e.g. a shuffled HF dataset), and a subset of rows
        other = ann.iloc[np.random.default_rng(5).permutation(len(ann))].reset_index(drop=True)
        assert _names_rows(_load(t, align_to=other), other)
        hf = Dataset.from_pandas(ann).shuffle(seed=3)  # has an indices mapping
        hf_tab = hf.to_pandas()
        assert _names_rows(_load(t, align_to=hf), hf_tab)
        sub = ann[ann['observation_id'] == 'obsC'].reset_index(drop=True)
        assert _names_rows(_load(t, align_to=sub), sub) and len(_load(t, align_to=sub)) == 10
        # values are exact copies of stored rows, not recomputed
        pos = {(o, f): i for i, (o, f) in enumerate(pd.read_parquet(emb_dir / 'row_keys.parquet').itertuples(index=False))}
        idx = [pos[(o, f)] for o, f in ann.itertuples(index=False)]
        assert torch.equal(out, emb[idx])
        # add_embeddings_from_disk aligns to the dataset it attaches to
        ds = add_embeddings_from_disk(Dataset.from_pandas(other), 's', 'v', 'enc', 'tok', dataset_root=str(t))
        assert _names_rows(torch.tensor(ds.with_format(None)[:]['embedding_enc_tok']), other)
        n_wrong = int((emb.numpy()[:, 1] != ann['frame_idx'].values).sum())
        print(f'    sidecar permuted: positional read would give {n_wrong}/30 wrong frames; reordered '
              f'output names the right frame for default / DataFrame / shuffled Dataset / subset / add_embeddings')


def test_d_missing_rows_error():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        ann, emb, emb_dir = _world(t, 'noB', with_sidecar=True)
        try:
            _load(t)
        except EmbeddingIndexError as e:
            msg = str(e)
            assert 'obsB (10 frames)' in msg and 'obsA' not in msg.split(':')[1], msg
        else:
            raise AssertionError('missing observation did not raise')
        # a table that only asks for cached observations is fine
        sub = ann[ann['observation_id'] != 'obsB'].reset_index(drop=True)
        assert _names_rows(_load(t, align_to=sub), sub)
        # sidecar row count != embedding row count
        write_row_keys(emb_dir, ann['observation_id'], ann['frame_idx'])  # 30 keys, 20 rows
        try:
            _load(t)
        except EmbeddingIndexError as e:
            assert 'does not describe this cache' in str(e)
        else:
            raise AssertionError('row-count mismatch did not raise')
        # sidecar present but no table to check it against
        (t / 's' / 'v' / 'annotations.csv').unlink()
        write_row_keys(emb_dir, sub['observation_id'], sub['frame_idx'])
        try:
            _load(t)
        except EmbeddingIndexError as e:
            assert 'align_to' in str(e)
        else:
            raise AssertionError('sidecar without a frame table did not raise')
        print(f'    missing rows -> "{msg[:110]}"')


# ----------------------------------------------------------------------------- real data
V1 = ROOT / 'dataset' / 'mice' / 'v1'


def test_e_real_mice_v1_class_l2():
    emb_dir = V1 / 'embeddings' / 'full' / 'dinov2' / 'class_l-2'
    if os.environ.get('EMB_KEYS_SKIP_REAL') == '1' or not (emb_dir / 'row_keys.parquet').exists():
        print('    SKIPPED (no real data)')
        return
    from build_row_keys import Recomputer
    ann = pd.read_csv(V1 / 'annotations.csv', usecols=['observation_id', 'frame_idx', 'frame_path'])
    t0 = time.time()
    old = ge._load_embeddings_positional(emb_dir, 'dinov2', 'class')  # the pre-change behaviour
    new = load_embeddings_from_disk('mice', 'v1', 'dinov2', 'class', layer=-2, dataset_root=str(ROOT / 'dataset'))
    assert new.shape == (len(ann), 768)
    # 6 rows >= 612,000 whose stored (cache-order) key differs from the annotations key there
    side = pd.read_parquet(emb_dir / 'row_keys.parquet')
    differ = np.flatnonzero((side['observation_id'].values != ann['observation_id'].values)
                            | (side['frame_idx'].values != ann['frame_idx'].values))
    assert differ.min() >= 612_000, differ.min()
    rows = differ[np.linspace(0, len(differ) - 1, 6).astype(int)]
    rec = Recomputer('cuda')
    fresh = rec('dinov2', 'cls', ann['frame_path'].values[rows].tolist())

    def cos(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    c_new, c_old = cos(new[rows].numpy(), fresh), cos(old[rows].numpy(), fresh)
    print(f'    mice v1 dinov2/class_l-2, annotations rows {list(rows)}:\n'
          f'      new loader cos {np.round(c_new, 5).tolist()}\n'
          f'      old positional cos {np.round(c_old, 4).tolist()}  [{time.time() - t0:.0f}s]')
    assert (c_new > 0.999).all(), c_new
    assert (c_old < 0.99).all(), c_old


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    if len(sys.argv) > 1:
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
