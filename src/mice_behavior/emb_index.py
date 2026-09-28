"""
Key-based lookup between annotations.csv rows and cached embedding rows.

Why this exists: cached embedding arrays (e.g. dinov2/class_l-2/embeddings.npy,
dinov2/patch_grid4/embeddings.npy) are written in whatever frame order the extraction
saw. When annotations.csv is later rebuilt (new observations, different ordering), the
Nth embedding row is no longer the Nth annotations row. Indexing by position then
returns the WRONG FRAME without any error (this happened for mice v1: the caches were
built from the 2,556,000-frame HF shards, annotations.csv was rebuilt on 2026-07-20 with
2,592,000 rows in a different order after row 611,999).

Contract: every cached embedding directory carries a sidecar ``row_keys.parquet`` with
columns (observation_id, frame_idx), one row per embedding row, in embedding row order.
All loaders resolve annotations rows -> embedding rows by joining on those keys. There is
no positional fallback: a missing sidecar, duplicated keys, or requested observations with
no embeddings are hard errors.

Public API:
    write_row_keys(emb_dir, observation_id, frame_idx)       -> Path
    load_row_keys(emb_dir)                                     -> DataFrame
    resolve_rows(emb_dir, annotations)                         -> int64 array, ann row -> emb row (-1 absent)
    require_rows(resolved, annotations, rows=None, obs_ids=None)  raises naming missing observations
    available_observations(emb_dir, annotations)               -> set of fully-covered observation ids
    gather_rows(array, rows)                                   -> np.ndarray (memmap-friendly fancy index)
    n_embedding_rows(emb_dir), cls_emb_dim(embeddings_path)
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROW_KEYS_FILE = 'row_keys.parquet'
KEY_COLS = ['observation_id', 'frame_idx']


class EmbeddingIndexError(RuntimeError):
    """Raised whenever cached embeddings cannot be matched to annotations rows by key."""


def _emb_dir(path) -> Path:
    """Accept either an embedding directory or a file inside it (embeddings.npy, global_idx.npy)."""
    p = Path(path)
    return p.parent if p.suffix else p


def write_row_keys(emb_dir, observation_id, frame_idx) -> Path:
    """Write <emb_dir>/row_keys.parquet: row i = key of embedding row i. Refuses duplicate keys."""
    emb_dir = _emb_dir(emb_dir)
    df = pd.DataFrame({
        'observation_id': np.asarray(observation_id).astype(str),
        'frame_idx': np.asarray(frame_idx).astype(np.int64),
    })
    dup = df.duplicated(KEY_COLS)
    if dup.any():
        raise EmbeddingIndexError(
            f'refusing to write {emb_dir / ROW_KEYS_FILE}: {int(dup.sum()):,} duplicated '
            f'(observation_id, frame_idx) keys, e.g. {df[dup].iloc[0].to_dict()}')
    out = emb_dir / ROW_KEYS_FILE
    tmp = out.with_suffix('.parquet.tmp')
    df.to_parquet(tmp, index=False)
    tmp.rename(out)
    _load_row_keys_cached.cache_clear()
    return out


def hf_column(dataset, name) -> np.ndarray:
    """One column of a HuggingFace Dataset as a numpy array, in dataset row order
    (honours an indices mapping from select/shuffle if present)."""
    if getattr(dataset, '_indices', None) is None:
        return dataset.data.column(name).to_numpy(zero_copy_only=False)
    return np.asarray(list(dataset[name]))


def _missing_sidecar_msg(emb_dir: Path) -> str:
    return (
        f'{emb_dir / ROW_KEYS_FILE} not found. Cached embeddings cannot be matched to '
        f'annotations.csv rows without it (positional indexing is unsafe: annotations.csv may '
        f'have been rebuilt in a different order since extraction).\n'
        f'  - For caches extracted from the current frame table, re-run the extractor '
        f'(src/embedding/get_embeddings.py or scripts/mice_behavior/extract_patch_embeddings.py) '
        f'which now writes {ROW_KEYS_FILE}.\n'
        f'  - For the legacy mice v1 caches, run: python scripts/mice_behavior/build_row_keys.py '
        f'(derives keys from dataset/mice/v1/eci/diagnostics/cls_l2_ann_row.npy) and verify them '
        f'with --verify.')


def _file_sig(p: Path):
    st = p.stat()
    return (str(p.resolve()), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=16)
def _load_row_keys_cached(sig):
    path = Path(sig[0])
    df = pd.read_parquet(path)
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise EmbeddingIndexError(f'{path} lacks columns {missing} (has {list(df.columns)})')
    df = df[KEY_COLS].astype({'observation_id': str, 'frame_idx': np.int64}).reset_index(drop=True)
    dup = df.duplicated(KEY_COLS)
    if dup.any():
        raise EmbeddingIndexError(
            f'{path} has {int(dup.sum()):,} duplicated (observation_id, frame_idx) keys, '
            f'e.g. {df[dup].iloc[0].to_dict()} -- cannot resolve embedding rows unambiguously')
    return df


def load_row_keys(emb_dir) -> pd.DataFrame:
    emb_dir = _emb_dir(emb_dir)
    path = emb_dir / ROW_KEYS_FILE
    if not path.exists():
        raise EmbeddingIndexError(_missing_sidecar_msg(emb_dir))
    return _load_row_keys_cached(_file_sig(path))


def n_embedding_rows(emb_dir) -> int:
    return len(load_row_keys(emb_dir))


def cls_emb_dim(embeddings_path, bytes_per_value: int = 4) -> int:
    """Embedding width of a headerless (n_rows, D) memmap, using the sidecar's row count
    (NOT len(annotations.csv), which differs from the cache's row count)."""
    p = Path(embeddings_path)
    n = n_embedding_rows(p)
    size = p.stat().st_size
    if size % (bytes_per_value * n):
        raise EmbeddingIndexError(
            f'{p}: {size:,} bytes is not a multiple of {n:,} rows x {bytes_per_value} bytes -- '
            f'{ROW_KEYS_FILE} does not describe this array')
    return size // (bytes_per_value * n)


@lru_cache(maxsize=4)
def _read_annotations_cached(sig):
    return pd.read_csv(sig[0], usecols=KEY_COLS)


def read_annotation_keys(annotations) -> pd.DataFrame:
    """annotations: path to annotations.csv or a DataFrame with observation_id/frame_idx.
    Row order is preserved; the returned frame's positional index = annotations row."""
    if isinstance(annotations, (str, Path)):
        return _read_annotations_cached(_file_sig(Path(annotations)))
    missing = [c for c in KEY_COLS if c not in annotations.columns]
    if missing:
        raise EmbeddingIndexError(f'annotations table lacks columns {missing}')
    return annotations


def resolve_rows(emb_dir, annotations, n_rows: int = None) -> np.ndarray:
    """Return int64 array r with len(annotations): r[a] = embedding row holding annotations
    row a's frame, or -1 if that frame has no cached embedding. Joins on (observation_id,
    frame_idx). n_rows (optional) = first dim of the embedding array, checked against the
    sidecar length."""
    emb_dir = _emb_dir(emb_dir)
    keys = load_row_keys(emb_dir)
    if n_rows is not None and n_rows != len(keys):
        raise EmbeddingIndexError(
            f'{emb_dir}: embedding array has {n_rows:,} rows but {ROW_KEYS_FILE} has {len(keys):,}')
    if isinstance(annotations, (str, Path)):  # memoised: loaders call this once per obs_boundary
        out = _resolve_cached(_file_sig(emb_dir / ROW_KEYS_FILE), _file_sig(Path(annotations)))
        return out
    return _join(keys, read_annotation_keys(annotations), emb_dir)


@lru_cache(maxsize=8)
def _resolve_cached(keys_sig, ann_sig):
    out = _join(_load_row_keys_cached(keys_sig), _read_annotations_cached(ann_sig), Path(keys_sig[0]).parent)
    out.setflags(write=False)
    return out


def _join(keys: pd.DataFrame, ann: pd.DataFrame, emb_dir) -> np.ndarray:
    left = pd.DataFrame({
        'observation_id': ann['observation_id'].astype(str).values,
        'frame_idx': ann['frame_idx'].astype(np.int64).values,
    })
    right = keys.assign(_emb_row=np.arange(len(keys), dtype=np.int64))
    merged = left.merge(right, on=KEY_COLS, how='left', sort=False)
    if len(merged) != len(left):  # only possible if right had duplicates, which load_row_keys forbids
        raise EmbeddingIndexError(f'{emb_dir}: key join changed row count ({len(left)} -> {len(merged)})')
    return merged['_emb_row'].fillna(-1).to_numpy(dtype=np.int64)


def require_rows(resolved: np.ndarray, annotations, rows=None, obs_ids=None, where: str = ''):
    """Raise EmbeddingIndexError naming every observation that has annotations rows (restricted
    to `rows` or to `obs_ids` if given) without a cached embedding."""
    ann = read_annotation_keys(annotations)
    oid = ann['observation_id'].to_numpy()
    if rows is not None:
        rows = np.asarray(rows, dtype=np.int64)
        bad = rows[resolved[rows] < 0]
    else:
        mask = resolved < 0
        if obs_ids is not None:
            mask &= np.isin(oid, np.asarray(list(obs_ids), dtype=object))
        bad = np.flatnonzero(mask)
    if len(bad):
        counts = pd.Series(oid[bad]).value_counts().sort_index()
        listing = ', '.join(f'{o} ({n} frames)' for o, n in counts.items())
        raise EmbeddingIndexError(
            f'{len(counts)} requested observation(s) have no cached embeddings{(" in " + where) if where else ""}: '
            f'{listing}. Drop them from obs_ids (see emb_index.available_observations) or extract '
            f'embeddings for them.')


def available_observations(emb_dir, annotations) -> set:
    """Observations whose every annotations row has a cached embedding in emb_dir."""
    ann = read_annotation_keys(annotations)
    resolved = resolve_rows(emb_dir, ann)
    ok = pd.Series(resolved >= 0).groupby(ann['observation_id'].to_numpy()).all()
    return set(ok.index[ok.values])


def gather_rows(array, rows) -> np.ndarray:
    """array[rows] copied into RAM, reading a memmap in ascending row order (sequential I/O),
    then restoring the requested order. All rows must be >= 0."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and rows.min() < 0:
        raise EmbeddingIndexError('gather_rows got unresolved (-1) rows')
    uniq, inv = np.unique(rows, return_inverse=True)
    return np.asarray(array[uniq])[inv]


def check_obs_boundary(annotations, obs_boundary: dict, where: str = ''):
    """Sanity-check that each {obs_id: (start, end)} range really covers exactly that
    observation in the annotations table the loader resolves against (guards against a
    caller that built obs_boundary from a different annotations.csv)."""
    ann = read_annotation_keys(annotations)
    oid = ann['observation_id'].to_numpy()
    for o, (s, e) in obs_boundary.items():
        if s < 0 or e > len(oid) or e <= s or not (oid[s] == o and oid[e - 1] == o) \
                or not (oid[s:e] == o).all():
            raise EmbeddingIndexError(
                f'obs_boundary[{o!r}] = ({s}, {e}) does not match the annotations table used for '
                f'embedding lookup{(" in " + where) if where else ""} -- caller and loader must use '
                f'the same annotations.csv')
