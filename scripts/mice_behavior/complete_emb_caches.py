"""
Complete the legacy mice v1 embedding caches by APPENDING the frames they lack.

Why: dinov2/dinov3 class_l-2 were extracted from the old 2,556,000-frame HF shards (no pool
rd64), and dinov2/dinov3 patch_grid4 from the 129 observations annotated at the time. The
rebuilt annotations.csv (2026-07-20) has 2,592,000 rows and pair_labels.parquet names 144
annotated observations. This script extracts ONLY the missing frames, with exactly the
original settings, and appends them after the existing rows (existing rows are never
rewritten):

    class_l-2  : EmbeddingExtractor(token='class', layer=-2) from src/embedding/get_embeddings.py
                 (AutoImageProcessor use_fast=True, fp16 autocast on GPU, float32 storage,
                 batch 192 as in logs/get_embeddings_60865676.out / 62361129.out).
                 Target: every annotations.csv frame.
    patch_grid4: patch_grid_forward() from scripts/mice_behavior/extract_patch_embeddings.py
                 (last_hidden_state minus CLS/registers, adaptive-avg-pool 4x4, fp16 autocast,
                 fp16 storage, batch 32). Target: every frame of an annotated observation.

Frames are read from the current HF frame dataset (00238 shards), which matches
annotations.csv row for row (checked at start-up on all rows).

Modes:
    --mode check   re-extract --n-check rows that ARE in the cache (resolved by key through
                   row_keys.parquet) and print cosine vs the stored rows. --rows new picks them
                   from the appended tail instead (use after --mode extract).
    --mode extract compute the missing rows into <cache>/append.npy.tmp, check them (finite,
                   no all-zero rows), back up row_keys.parquet / global_idx.npy to *.bak, append
                   the bytes to embeddings.npy, then atomically swap in the extended
                   row_keys.parquet (and global_idx.npy for patch_grid4, whose appended entries
                   are CURRENT annotations.csv rows; loaders only use its length) and, for
                   class_l-2, regenerate embeddings.pt from the new embeddings.npy.

Usage:
    python scripts/mice_behavior/complete_emb_caches.py --encoder dinov2 --kind cls --mode check
    python scripts/mice_behavior/complete_emb_caches.py --encoder dinov2 --kind cls --mode extract
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'mice_behavior'))
import extract_patch_embeddings as xpe  # noqa: E402  (also installs the PIL ExifTags shim)
from src.dataset.get_dataset import load_dataset  # noqa: E402
from src.embedding.get_embeddings import EmbeddingExtractor  # noqa: E402
from src.mice_behavior.emb_index import (  # noqa: E402
    KEY_COLS, hf_column, load_row_keys, resolve_rows, write_row_keys)

V1 = ROOT / 'dataset' / 'mice' / 'v1'
EMB = V1 / 'embeddings' / 'full'
EMB_DIM = 768
KINDS = {  # kind -> (cache subdir, row shape, storage dtype, original batch size)
    'cls': ('class_l-2', (EMB_DIM,), np.float32, 192),
    'patch': ('patch_grid4', (16, EMB_DIM), np.float16, 32),
}


def cosine(a, b):
    a, b = a.reshape(len(a), -1).astype(np.float64), b.reshape(len(b), -1).astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


class _Frames(torch.utils.data.Dataset):
    def __init__(self, ds, rows, proc):
        self.ds, self.rows, self.proc = ds, rows, proc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        out = self.proc(images=self.ds[int(self.rows[i])]['image'], return_tensors='pt')
        return {k: v.squeeze(0) for k, v in out.items()}


class Encoder:
    """Exactly the original forward passes (see module docstring)."""

    def __init__(self, encoder, kind, batch_size, num_workers, device='cuda'):
        self.kind, self.batch_size, self.num_workers = kind, batch_size, num_workers
        if kind == 'cls':
            self.ext = EmbeddingExtractor(encoder=encoder, device=device, batch_size=batch_size,
                                          num_workers=num_workers, token='class', layer=-2)
            self.proc = self.ext.processor
        else:
            from transformers import AutoImageProcessor, AutoModel
            self.device = torch.device(device)
            self.proc = AutoImageProcessor.from_pretrained(xpe.MODEL_IDS[encoder], use_fast=True)
            self.model = AutoModel.from_pretrained(xpe.MODEL_IDS[encoder]).to(self.device)
            self.model.eval()
            self.model.requires_grad_(False)
            self.n_prefix = 1 + getattr(self.model.config, 'num_register_tokens', 0)
            self.autocast = torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                           enabled=self.device.type == 'cuda')

    def _forward(self, pixel_values):
        if self.kind == 'cls':
            return self.ext._forward_tensors(pixel_values).numpy()
        return xpe.patch_grid_forward(self.model, pixel_values.to(self.device), self.n_prefix, 4,
                                      self.autocast).cpu().numpy()

    def run(self, hf, rows, out, dtype):
        """Fill out[i] with the (storage-dtype) feature of HF row rows[i]."""
        loader = torch.utils.data.DataLoader(
            _Frames(hf, rows, self.proc), batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True, shuffle=False, prefetch_factor=4 if self.num_workers else None)
        cur, t0 = 0, time.time()
        for nb, b in enumerate(loader, 1):
            f = self._forward(b['pixel_values']).astype(dtype)
            out[cur:cur + len(f)] = f
            cur += len(f)
            if nb % 50 == 0:
                print(f'    {cur:,}/{len(rows):,} frames  ({cur / (time.time() - t0):.0f} frames/s)', flush=True)
        assert cur == len(rows), (cur, len(rows))
        return out


def target_rows(kind, ann):
    """annotations.csv rows every complete cache of this kind must cover."""
    if kind == 'cls':
        return np.arange(len(ann), dtype=np.int64)
    annotated = set(pd.read_parquet(V1 / 'pair_labels.parquet')['observation_id'].unique())
    return np.flatnonzero(ann['observation_id'].isin(annotated).to_numpy()).astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', required=True, choices=sorted(xpe.MODEL_IDS))
    p.add_argument('--kind', required=True, choices=sorted(KINDS))
    p.add_argument('--mode', required=True, choices=['check', 'extract'])
    p.add_argument('--rows', default='old', choices=['old', 'new'], help='check mode: sample stored rows from the legacy prefix or the appended tail')
    p.add_argument('--n-check', type=int, default=20)
    p.add_argument('--n-legacy', type=int, default=None, help='rows before the appended tail (default: current length for --rows old)')
    p.add_argument('--num-workers', type=int, default=8)
    args = p.parse_args()

    subdir, row_shape, dtype, batch_size = KINDS[args.kind]
    emb_dir = EMB / args.encoder / subdir
    emb_path = emb_dir / 'embeddings.npy'
    row_bytes = int(np.prod(row_shape)) * np.dtype(dtype).itemsize

    ann = pd.read_csv(V1 / 'annotations.csv', usecols=KEY_COLS)
    hf = load_dataset(subject='mice', version='v1', dataset_root=str(ROOT / 'dataset'), frame_type='full')
    if len(hf) != len(ann) or not (
            (hf_column(hf, 'observation_id') == ann['observation_id'].to_numpy()).all()
            and (hf_column(hf, 'frame_idx').astype(np.int64) == ann['frame_idx'].to_numpy()).all()):
        raise RuntimeError('HF frame dataset and annotations.csv disagree on (observation_id, frame_idx)')
    print(f'HF frame dataset == annotations.csv on all {len(ann):,} keys')

    keys = load_row_keys(emb_dir)
    n_rows = len(keys)
    if emb_path.stat().st_size != n_rows * row_bytes:
        raise RuntimeError(f'{emb_path}: {emb_path.stat().st_size:,} bytes != {n_rows:,} rows x {row_bytes} B')
    resolved = resolve_rows(emb_dir, ann, n_rows=n_rows)  # ann row -> emb row
    stored = np.memmap(emb_path, dtype=dtype, mode='r', shape=(n_rows, *row_shape))
    enc = Encoder(args.encoder, args.kind, batch_size, args.num_workers)
    print(f'{args.encoder}/{subdir}: {n_rows:,} rows, shape {stored.shape}, dtype {stored.dtype}')

    if args.mode == 'check':
        lo = 0 if args.rows == 'old' else int(args.n_legacy)
        emb_rows = np.unique(np.linspace(lo, n_rows - 1, args.n_check).astype(np.int64))
        inv = np.full(n_rows, -1, dtype=np.int64)
        ok = resolved >= 0
        inv[resolved[ok]] = np.flatnonzero(ok)
        ann_rows = inv[emb_rows]  # the annotations row whose key the sidecar gives this emb row
        assert (ann_rows >= 0).all()
        fresh = enc.run(hf, ann_rows, np.empty((len(ann_rows), *row_shape), dtype=np.float32), np.float32)
        cos = cosine(np.asarray(stored[emb_rows], dtype=np.float32), fresh)
        for r, a, c in zip(emb_rows, ann_rows, cos):
            print(f'  emb row {r:>9,} -> {ann["observation_id"].iat[a]} frame {ann["frame_idx"].iat[a]:>5}: cos {c:.6f}')
        print(f'CHECK {args.encoder}/{subdir} rows={args.rows}: n={len(cos)} min cos {cos.min():.6f} '
              f'mean {cos.mean():.6f} -> {"PASS" if cos.min() >= 0.9999 else "FAIL"}')
        sys.exit(0 if cos.min() >= 0.9999 else 1)

    # ---- extract ----
    tgt = target_rows(args.kind, ann)
    missing = tgt[resolved[tgt] < 0]
    print(f'target {len(tgt):,} annotations rows, {len(missing):,} missing '
          f'({ann["observation_id"].iloc[missing].nunique()} observations: '
          f'{sorted(ann["observation_id"].iloc[missing].unique())})')
    if not len(missing):
        print('nothing to do')
        return
    tmp = emb_dir / 'append.npy.tmp'
    new = np.memmap(tmp, dtype=dtype, mode='w+', shape=(len(missing), *row_shape))
    t0 = time.time()
    enc.run(hf, missing, new, dtype)
    new.flush()
    print(f'extracted {len(missing):,} frames in {time.time() - t0:.0f}s')
    flat = np.asarray(new, dtype=np.float32).reshape(len(missing), -1)
    n_bad = int((~np.isfinite(flat)).any(1).sum()); n_zero = int((np.abs(flat).sum(1) == 0).sum())
    if n_bad or n_zero:
        raise RuntimeError(f'new rows: {n_bad} non-finite, {n_zero} all-zero -- not appending')
    del flat

    # backups of the small metadata files, extended versions written next to them
    import shutil
    new_oid = np.concatenate([keys['observation_id'].to_numpy(), ann['observation_id'].to_numpy()[missing]])
    new_fidx = np.concatenate([keys['frame_idx'].to_numpy(), ann['frame_idx'].to_numpy()[missing]])
    shutil.copy2(emb_dir / 'row_keys.parquet', emb_dir / 'row_keys.parquet.bak')
    gi_path = emb_dir / 'global_idx.npy'
    if gi_path.exists():
        shutil.copy2(gi_path, emb_dir / 'global_idx.npy.bak')
        gi = np.load(gi_path)
        assert len(gi) == n_rows
        np.save(emb_dir / 'global_idx.tmp.npy', np.concatenate([gi, missing.astype(gi.dtype)]))

    # append bytes (existing rows untouched), then swap the metadata in
    size0 = emb_path.stat().st_size
    with open(emb_path, 'r+b') as f, open(tmp, 'rb') as src:
        f.seek(size0)
        shutil.copyfileobj(src, f, length=64 << 20)
        f.flush(); os.fsync(f.fileno())
    n_new = n_rows + len(missing)
    assert emb_path.stat().st_size == n_new * row_bytes, 'append size mismatch'
    write_row_keys(emb_dir, new_oid, new_fidx)  # tmp + rename; refuses duplicate keys
    if gi_path.exists():
        os.replace(emb_dir / 'global_idx.tmp.npy', gi_path)
    tmp.unlink()
    print(f'appended: {emb_path} now {n_new:,} rows ({emb_path.stat().st_size:,} bytes)')

    pt = emb_dir / 'embeddings.pt'
    if pt.exists():
        arr = np.array(np.memmap(emb_path, dtype=np.float32, mode='r', shape=(n_new, EMB_DIM)))
        torch.save(torch.from_numpy(arr), emb_dir / 'embeddings.pt.tmp')
        os.replace(emb_dir / 'embeddings.pt.tmp', pt)
        print(f'regenerated {pt} from embeddings.npy ({n_new:,} x {EMB_DIM})')


if __name__ == '__main__':
    main()
