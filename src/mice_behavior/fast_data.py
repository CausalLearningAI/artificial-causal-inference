"""
Fully vectorized batch construction for the mouse pairwise behavior classifier.

PyTorch's Dataset/DataLoader calls __getitem__ once per individual sample
regardless of batch_size or worker count, with each call doing its own dict
lookup + numpy slice + tensor construction in Python — a per-call cost that
scales with total sample count, not batch count.

This module avoids that entirely: pad each observation's preloaded embedding
block with `context_k` zero-rows on both ends and concatenate all observations
into one flat array. Every sample's context window is then a fixed length
(2*context_k+1) at a precomputed flat-array offset, so an entire batch's worth
of windows can be fetched with a single vectorized numpy fancy-index gather.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class FastBatchData:
    """Precomputes a flat, padded embedding array + vectorized index/mask arrays
    for O(1)-Python-overhead batch construction.

    Works for both CLS (emb shape (N, D)) and patch-grid (emb shape (N, P, D))
    inputs — pass whichever preloaded array you have.
    """

    def __init__(
        self,
        annotations_csv: str,
        pair_labels_parquet: str,
        obs_ids,
        context_k: int,
        emb_dim: int,
        load_embeddings_fn,
        n_patches: int = None,
    ):
        """
        load_embeddings_fn(obs_boundary: dict[str, (int,int)]) -> dict[int, np.ndarray]
            Given {obs_id: (global_start, global_end)}, return {global_start: array}
            where array has shape (obs_len, emb_dim) or (obs_len, n_patches, emb_dim),
            already preloaded into RAM (this function owns the actual embedding I/O —
            CLS vs patch-grid loading differs, so it's injected rather than duplicated
            here).
        """
        self.k = context_k
        self.emb_dim = emb_dim
        self.n_patches = n_patches

        ann = pd.read_csv(annotations_csv, usecols=['observation_id', 'frame_idx'])
        obs_set = set(obs_ids) if obs_ids is not None else None
        if obs_set is not None:
            ann = ann[ann['observation_id'].isin(obs_set)]
        ann_reset = ann.reset_index()
        obs_boundary = {}
        for oid, grp in ann_reset.groupby('observation_id', sort=False):
            idx = grp['index'].values
            obs_boundary[oid] = (int(idx[0]), int(idx[-1]) + 1)

        pair_labels = pd.read_parquet(pair_labels_parquet)
        if obs_set is not None:
            pair_labels = pair_labels[pair_labels['observation_id'].isin(obs_set)]
        frame_to_global = ann_reset[['observation_id', 'frame_idx', 'index']].rename(
            columns={'index': 'global_idx'}
        )
        pair_labels = pair_labels.merge(
            frame_to_global, on=['observation_id', 'frame_idx'], how='inner'
        ).astype({'global_idx': np.int32})

        annotated_obs = set(pair_labels['observation_id'].unique())
        ann_annotated = ann_reset[ann_reset['observation_id'].isin(annotated_obs)]

        PAIRS = np.array([(a1, a2) for a1 in range(4) for a2 in range(4) if a1 != a2], dtype=np.int8)
        a1_vals, a2_vals = PAIRS[:, 0], PAIRS[:, 1]

        all_global, all_a1, all_a2, all_obs_s, all_obs_e = [], [], [], [], []
        for oid, grp in ann_annotated.groupby('observation_id', sort=False):
            global_idxs = grp['index'].values.astype(np.int32)
            n = len(global_idxs)
            obs_s, obs_e = obs_boundary[oid]
            all_global.append(np.repeat(global_idxs, 12))
            all_a1.append(np.tile(a1_vals, n))
            all_a2.append(np.tile(a2_vals, n))
            all_obs_s.append(np.full(n * 12, obs_s, dtype=np.int32))
            all_obs_e.append(np.full(n * 12, obs_e, dtype=np.int32))

        gi = np.concatenate(all_global)
        a1 = np.concatenate(all_a1)
        a2 = np.concatenate(all_a2)
        obs_s_arr = np.concatenate(all_obs_s)
        obs_e_arr = np.concatenate(all_obs_e)

        pos = (
            pair_labels[['global_idx', 'agent1', 'agent2', 'label']]
            .rename(columns={'agent1': 'a1', 'agent2': 'a2'})
            .astype({'global_idx': np.int32, 'a1': np.int32, 'a2': np.int32})
        )
        samples_df = pd.DataFrame(
            {'global_idx': gi.astype(np.int32), 'a1': a1.astype(np.int32), 'a2': a2.astype(np.int32)}
        )
        merged = samples_df.merge(pos, on=['global_idx', 'a1', 'a2'], how='left')
        labels = merged['label'].fillna(0).values.astype(np.int8)

        # --- Build flat padded embedding array ---
        print('  loading embeddings for vectorized batching...')
        obs_arrays = load_embeddings_fn(obs_boundary)  # {obs_s: (obs_len, ...) array}
        k = context_k
        pad_start_by_obs_s = {}
        blocks = []
        cursor = 0
        for obs_s in sorted(obs_arrays.keys()):
            arr = obs_arrays[obs_s]
            obs_len = arr.shape[0]
            pad_shape = (k,) + arr.shape[1:]
            padded = np.concatenate([np.zeros(pad_shape, dtype=arr.dtype), arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)
            blocks.append(padded)
            pad_start_by_obs_s[obs_s] = cursor
            cursor += padded.shape[0]
        self.flat = np.concatenate(blocks, axis=0)  # (total_padded_rows, ...)
        del blocks

        local = gi - obs_s_arr
        obs_len_arr = obs_e_arr - obs_s_arr
        pad_start_arr = np.array([pad_start_by_obs_s[int(s)] for s in obs_s_arr], dtype=np.int64)
        # window for sample i: flat[pad_start_i + local_i : pad_start_i + local_i + 2k+1]
        # (padding absorbs boundary cases — no per-sample branching needed at all)
        self.centers = pad_start_arr + local  # start index of each sample's fixed-length window
        offsets_grid = np.arange(-k, k + 1, dtype=np.int64)  # (T,)
        self.offsets_grid = offsets_grid
        T = len(offsets_grid)
        abs_pos = local[:, None] + offsets_grid[None, :]  # (N, T)
        self.pad_mask = (abs_pos < 0) | (abs_pos >= obs_len_arr[:, None])  # True = padding position

        self.a1 = a1.astype(np.int64)
        self.a2 = a2.astype(np.int64)
        self.labels = labels.astype(np.int64)

        unique, counts = np.unique(labels, return_counts=True)
        print(f'  {len(labels):,} samples | ' + ' '.join(f'label{c}={cnt:,}' for c, cnt in zip(unique, counts)))

    def __len__(self):
        return len(self.labels)

    def to_device(self, dev):
        """Move the flat embedding array + small index arrays onto GPU once, so
        get_batch's gather runs via intra-GPU memory bandwidth instead of repeated
        host->device transfers. self.device is only set after every tensor is moved
        successfully — on OOM, local variables are discarded and the object is
        left in its original CPU/numpy state rather than half-initialized."""
        flat_t = torch.from_numpy(self.flat).to(dev)
        centers_t = torch.from_numpy(self.centers).to(dev)
        pad_mask_t = torch.from_numpy(self.pad_mask).to(dev)
        a1_t = torch.from_numpy(self.a1).to(dev)
        a2_t = torch.from_numpy(self.a2).to(dev)
        labels_t = torch.from_numpy(self.labels).to(dev)
        offsets_grid_t = torch.from_numpy(self.offsets_grid).to(dev)
        self.flat_t, self.centers_t, self.pad_mask_t = flat_t, centers_t, pad_mask_t
        self.a1_t, self.a2_t, self.labels_t = a1_t, a2_t, labels_t
        self.offsets_grid_t = offsets_grid_t
        self.device = dev
        return self

    def get_batch(self, idx):
        """Vectorized fetch — one gather for the whole batch, no Python per-sample loop.
        If to_device() was called, idx may be a numpy array or GPU long tensor; the gather
        and all outputs stay resident on GPU (no per-batch host->device transfer needed)."""
        if getattr(self, 'device', None) is not None:
            idx_t = idx if torch.is_tensor(idx) else torch.from_numpy(idx).to(self.device)
            T = len(self.offsets_grid_t)
            window_idx = self.centers_t[idx_t].unsqueeze(1) + torch.arange(T, device=self.device).unsqueeze(0)
            context = self.flat_t[window_idx].float()
            mask = self.pad_mask_t[idx_t]
            offs = self.offsets_grid_t.unsqueeze(0).expand(len(idx_t), T)
            return context, offs, self.a1_t[idx_t], self.a2_t[idx_t], self.labels_t[idx_t], mask

        T = len(self.offsets_grid)
        window_idx = self.centers[idx][:, None] + np.arange(T)[None, :]  # (B, T)
        context = self.flat[window_idx]  # (B, T, ...) — single fancy-index gather, native dtype
        mask = self.pad_mask[idx]  # (B, T)
        offs = np.broadcast_to(self.offsets_grid, (len(idx), T))
        # Keep native dtype (fp16 for patch-grid) instead of upcasting to fp32 here — that
        # would double the CPU gather/copy volume AND the host->device PCIe transfer size
        # for no benefit; the caller upcasts after .to(dev), where fp16->fp32 is nearly free
        # (GPU memory bandwidth, not the much slower PCIe bus).
        return (
            torch.from_numpy(np.ascontiguousarray(context)),
            torch.from_numpy(np.ascontiguousarray(offs)).long(),
            torch.from_numpy(self.a1[idx]),
            torch.from_numpy(self.a2[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(np.ascontiguousarray(mask)),
        )


def load_cls_embeddings(embeddings_path: str, emb_dim: int):
    def _load(obs_boundary):
        emb_path = Path(embeddings_path)
        n_total = emb_path.stat().st_size // (4 * emb_dim)
        mmap = np.memmap(emb_path, dtype='float32', mode='r', shape=(n_total, emb_dim))
        return {obs_s: np.array(mmap[obs_s:obs_e]) for obs_s, obs_e in obs_boundary.values()}
    return _load


def load_patchgrid_embeddings(embeddings_path: str, global_idx_path: str, n_patches: int, emb_dim: int):
    def _load(obs_boundary):
        global_idx = np.load(global_idx_path)
        row_of_global = {int(g): i for i, g in enumerate(global_idx)}
        mmap = np.memmap(embeddings_path, dtype='float16', mode='r', shape=(len(global_idx), n_patches, emb_dim))
        out = {}
        for obs_s, obs_e in obs_boundary.values():
            pg_start = row_of_global[obs_s]
            pg_end = row_of_global[obs_e - 1] + 1
            out[obs_s] = np.array(mmap[pg_start:pg_end])
        return out
    return _load
