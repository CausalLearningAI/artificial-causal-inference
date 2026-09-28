"""
Read access to stored DINOv2 patch tokens for SAE training / evaluation (ECI).

Two on-disk layouts are supported:
    single   <root>/patch_tokens.npy (N, P, d), cls.npy, metadata.parquet      (v1, 64 frames/obs)
    sharded  <root>/shards/shard_XX/{patch_tokens.npy, cls.npy, metadata.parquet, DONE}
             + <root>/metadata.parquet (all shards concatenated, columns shard, shard_row)
             (v2, 1 fps; frames stored in a seeded random order, so any contiguous block
             of rows is a uniform random sample of frames over all observations)

Rows are read with file.readinto (releases the GIL, so a background thread can prefetch
the next chunk while the GPU trains on the current one).

Classes / functions:
    TokenStore            meta, n_frames, frames(idx), iter_chunks(mask, ...), cls(idx)
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


class _NpyRows:
    """Row reader for a 3-d or 2-d float16 .npy file via readinto."""

    def __init__(self, path):
        mm = np.load(path, mmap_mode='r')
        self.path, self.shape, self.dtype, self.offset = str(path), mm.shape, mm.dtype, mm.offset
        self.row_elems = int(np.prod(mm.shape[1:]))
        self.row_bytes = self.row_elems * mm.dtype.itemsize
        del mm

    def read(self, lo, hi, out=None):
        """Rows [lo, hi) -> array (hi-lo, *shape[1:])."""
        n = hi - lo
        if out is None:
            out = np.empty((n,) + tuple(self.shape[1:]), dtype=self.dtype)
        assert out.flags.c_contiguous and out.dtype == self.dtype and out.shape[0] == n
        with open(self.path, 'rb', buffering=0) as f:
            f.seek(self.offset + lo * self.row_bytes)
            view = memoryview(out.reshape(-1).view(np.uint8))
            got = 0
            while got < n * self.row_bytes:
                k = f.readinto(view[got:])
                if not k:
                    raise IOError(f'short read in {self.path} rows [{lo}, {hi})')
                got += k
        return out


class TokenStore:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = pd.read_parquet(self.root / 'metadata.parquet')
        self.n_frames = len(self.meta)
        shard_root = self.root / 'shards'
        self.sharded = shard_root.is_dir()
        if self.sharded:
            n_shards = int(self.meta['shard'].max()) + 1
            dirs = [shard_root / f'shard_{i:02d}' for i in range(n_shards)]
            missing = [str(d) for d in dirs if not (d / 'DONE').exists()]
            if missing:
                raise RuntimeError(f'unfinished shards: {missing}')
            self._tok = [_NpyRows(d / 'patch_tokens.npy') for d in dirs]
            self._cls = [_NpyRows(d / 'cls.npy') for d in dirs]
            sizes = np.array([t.shape[0] for t in self._tok])
            self.starts = np.concatenate([[0], np.cumsum(sizes)])
            if self.starts[-1] != self.n_frames:
                raise RuntimeError(f'shards hold {self.starts[-1]} frames, metadata {self.n_frames}')
        else:
            self._tok = [_NpyRows(self.root / 'patch_tokens.npy')]
            self._cls = [_NpyRows(self.root / 'cls.npy')]
            self.starts = np.array([0, self.n_frames])
        self.n_patches, self.dim = self._tok[0].shape[1], self._tok[0].shape[2]
        cfg = self.root / 'config.json'
        self.config = json.loads(cfg.read_text()) if cfg.exists() else {}

    # ---------------------------------------------------------------- random access
    def _runs(self, idx):
        """Sorted global frame indices -> list of (shard, lo, hi, out_pos) contiguous runs."""
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) and np.any(np.diff(idx) <= 0):
            raise ValueError('idx must be strictly increasing')
        runs, i = [], 0
        while i < len(idx):
            j = i
            s = int(np.searchsorted(self.starts, idx[i], side='right') - 1)
            end = self.starts[s + 1]
            while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1 and idx[j + 1] < end:
                j += 1
            runs.append((s, int(idx[i] - self.starts[s]), int(idx[j] - self.starts[s] + 1), i))
            i = j + 1
        return runs

    def _gather(self, readers, idx, n_threads=8):
        runs = self._runs(idx)
        out = np.empty((len(idx),) + tuple(readers[0].shape[1:]), dtype=readers[0].dtype)

        def job(r):
            s, lo, hi, pos = r
            readers[s].read(lo, hi, out[pos:pos + hi - lo])

        with ThreadPoolExecutor(n_threads) as ex:
            list(ex.map(job, runs))
        return out

    def frames(self, idx, n_threads=8):
        """Patch tokens of global frames idx (sorted) -> (n, P, d) float16."""
        return self._gather(self._tok, idx, n_threads)

    def cls(self, idx, n_threads=8):
        return self._gather(self._cls, idx, n_threads)

    # ---------------------------------------------------------------- streaming
    def chunk_bounds(self, chunk_frames):
        """Fixed chunk boundaries [(lo, hi)] in global frame index, never crossing a shard."""
        out = []
        for s in range(len(self._tok)):
            a, b = int(self.starts[s]), int(self.starts[s + 1])
            out += [(lo, min(lo + chunk_frames, b)) for lo in range(a, b, chunk_frames)]
        return out

    def iter_chunks(self, frame_mask, chunk_frames=16384, order_seed=0, prefetch=True):
        """Yield (chunk_id, tokens (n_sel*P, d) float16) for every chunk, in a seeded random
        chunk order, keeping only frames where frame_mask is True. A background thread
        reads the next chunk while the caller consumes the current one."""
        bounds = self.chunk_bounds(chunk_frames)
        order = np.random.default_rng(order_seed).permutation(len(bounds))
        frame_mask = np.asarray(frame_mask, dtype=bool)

        def load(ci):
            lo, hi = bounds[ci]
            s = int(np.searchsorted(self.starts, lo, side='right') - 1)
            block = self._tok[s].read(lo - int(self.starts[s]), hi - int(self.starts[s]))
            keep = frame_mask[lo:hi]
            if not keep.all():
                block = block[keep]
            return block.reshape(-1, self.dim)

        if not prefetch:
            for ci in order:
                yield int(ci), load(ci)
            return
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(load, order[0])
            for n, ci in enumerate(order):
                block = fut.result()
                if n + 1 < len(order):
                    fut = ex.submit(load, order[n + 1])
                yield int(ci), block

    def n_selected_tokens_per_chunk(self, frame_mask, chunk_frames=16384):
        frame_mask = np.asarray(frame_mask, dtype=bool)
        return np.array([int(frame_mask[lo:hi].sum()) * self.n_patches
                         for lo, hi in self.chunk_bounds(chunk_frames)], dtype=np.int64)
