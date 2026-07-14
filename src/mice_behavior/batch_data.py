"""
Fully vectorized batch construction for the mouse behavior classifiers
(FastBatchData for the pairwise task, FastFrameData for the per-frame task).

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
        stride: int = 1,
        max_frames: int = None,
        seed: int = 42,
    ):
        """
        load_embeddings_fn(obs_boundary: dict[str, (int,int)]) -> dict[int, np.ndarray]
            Given {obs_id: (global_start, global_end)}, return {global_start: array}
            where array has shape (obs_len, emb_dim) or (obs_len, n_patches, emb_dim),
            already preloaded into RAM (this function owns the actual embedding I/O —
            CLS vs patch-grid loading differs, so it's injected rather than duplicated
            here).
        stride: context positions are context_k*stride..context_k*stride in steps of
            `stride` (still 2*context_k+1 positions total — same attention/compute cost
            as stride=1), reaching a wider temporal window (+-context_k*stride frames)
            without the O(window length) cost of a larger dense context_k.
        max_frames: bounds the total number of frames whose embeddings get loaded/padded
            into self.flat (None = unrestricted, every annotated frame of every obs_id).
            Every frame with >=1 positive (nt/nn) sample is always kept in full (plus its
            +-reach context); on top of that, a random sample of negative frames (+context)
            is added up to max_frames. Samples whose frame isn't kept are dropped entirely
            (this reduces the number of pair-samples too, not just memory footprint) — needed
            because patch-grid's per-frame footprint is 16x CLS's, so its full data doesn't
            fit as GPU-resident, but observation-level bounding doesn't help here since
            positive events are spread across nearly every observation's full frame span.
        """
        self.k = context_k
        self.stride = stride
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

        k = context_k
        reach = k * stride  # frames of padding needed on each side to cover the widest offset
        local = gi - obs_s_arr
        obs_len_arr = obs_e_arr - obs_s_arr

        if max_frames is not None:
            # Reduce to a bounded set of ANCHOR frames (every positive frame, plus a random
            # sample of negative frames up to max_frames total) before loading/padding
            # embeddings — every frame with >=1 positive sample is a "sample" whether we bound
            # or not, so we always predict on them; negative frames beyond the anchor budget
            # are simply never used as prediction targets this construction (still available in
            # full for other training runs/seeds). `kept` additionally covers each anchor's
            # +-reach context frames, which never themselves become samples — only used to
            # supply the anchor's context window.
            n_total = int(gi.max()) + 1
            kept = np.zeros(n_total, dtype=bool)
            is_anchor = np.zeros(n_total, dtype=bool)

            frame_df = pd.DataFrame({'gi': gi, 'obs_s': obs_s_arr, 'obs_e': obs_e_arr, 'label': labels})
            frame_info = frame_df.groupby('gi', sort=False).agg(
                obs_s=('obs_s', 'first'), obs_e=('obs_e', 'first'), has_pos=('label', lambda s: (s > 0).any()),
            )
            pos_frames = frame_info[frame_info['has_pos']]
            for f_gi, row in pos_frames.iterrows():
                lo, hi = max(row.obs_s, f_gi - reach), min(row.obs_e - 1, f_gi + reach)
                kept[lo:hi + 1] = True
                is_anchor[f_gi] = True
            total_kept = int(kept.sum())

            if total_kept < max_frames:
                neg_frames = frame_info[~frame_info['has_pos']]
                neg_gi = neg_frames.index.to_numpy()
                rng = np.random.default_rng(seed)
                rng.shuffle(neg_gi)
                obs_s_of = neg_frames['obs_s'].to_dict()
                obs_e_of = neg_frames['obs_e'].to_dict()
                for f_gi in neg_gi:
                    if total_kept >= max_frames:
                        break
                    if kept[f_gi]:
                        continue  # already covered as some anchor's context
                    lo, hi = max(obs_s_of[f_gi], f_gi - reach), min(obs_e_of[f_gi] - 1, f_gi + reach)
                    added = int((~kept[lo:hi + 1]).sum())
                    if added:
                        kept[lo:hi + 1] = True
                        is_anchor[f_gi] = True
                        total_kept += added
            print(f'  bounded train frames: kept {total_kept:,} of {n_total:,} annotated frames '
                  f'({int(is_anchor.sum()):,} anchor frames -> samples)')

            sample_keep = is_anchor[gi]
            gi, a1, a2, obs_s_arr, obs_e_arr, labels, local, obs_len_arr = (
                gi[sample_keep], a1[sample_keep], a2[sample_keep], obs_s_arr[sample_keep],
                obs_e_arr[sample_keep], labels[sample_keep], local[sample_keep], obs_len_arr[sample_keep],
            )

        # --- Build flat padded embedding array ---
        print('  loading embeddings for vectorized batching...')
        obs_arrays = load_embeddings_fn(obs_boundary)  # {obs_s: (obs_len, ...) array}
        blocks = []
        cursor = 0
        # center_base[obs_s][p] such that, for a sample at true local position p in obs_s,
        # centers = p + center_base[obs_s][p] gives the flat position of its offset=-reach frame.
        # Unbounded: one run per obs (center_base is a constant = pad_start - 0). Bounded: one run
        # per contiguous cluster of kept local positions (center_base varies by run).
        center_base_by_obs = {}
        for obs_s in sorted(obs_arrays.keys()):
            arr = obs_arrays[obs_s]
            obs_len = arr.shape[0]
            obs_e = obs_s + obs_len
            if max_frames is None:
                runs = [(0, obs_len - 1)]
            else:
                kept_local = np.where(kept[obs_s:obs_e])[0]
                if len(kept_local) == 0:
                    continue
                split_at = np.where(np.diff(kept_local) != 1)[0] + 1
                runs = [(int(r[0]), int(r[-1])) for r in np.split(kept_local, split_at)]

            center_base = np.zeros(obs_len, dtype=np.int64)
            for run_lo, run_hi in runs:
                sub = arr[run_lo:run_hi + 1]
                pad_shape = (reach,) + arr.shape[1:]
                padded = np.concatenate([np.zeros(pad_shape, dtype=arr.dtype), sub, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)
                blocks.append(padded)
                run_pad_start = cursor
                cursor += padded.shape[0]
                center_base[run_lo:run_hi + 1] = run_pad_start - run_lo
            center_base_by_obs[obs_s] = center_base
        self.flat = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, emb_dim) if n_patches is None else (0, n_patches, emb_dim), dtype=np.float32)
        del blocks

        center_adjust = np.empty(len(local), dtype=np.int64)
        for obs_s, center_base in center_base_by_obs.items():
            rows = np.where(obs_s_arr == obs_s)[0]
            if len(rows):
                center_adjust[rows] = center_base[local[rows]]

        # centers[i] + offsets_grid_local (0, stride, 2*stride, ..., 2*reach) indexes flat[]
        # at true offsets -reach..+reach in steps of `stride` (reach = k*stride).
        self.centers = local + center_adjust  # position of this sample's offset=-reach frame
        self.offsets_grid_local = np.arange(0, 2 * reach + 1, stride, dtype=np.int64)  # (T,) flat-array steps
        offsets_grid = self.offsets_grid_local - reach  # (T,) true frame offsets, e.g. -6..6 step 3
        self.offsets_grid = offsets_grid
        T = len(offsets_grid)
        abs_pos = local[:, None] + offsets_grid[None, :]  # (N, T)
        self.pad_mask = (abs_pos < 0) | (abs_pos >= obs_len_arr[:, None])  # True = padding position

        self.a1 = a1.astype(np.int64)
        self.a2 = a2.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.gi = gi.astype(np.int64)  # global frame index (row into annotations.csv) per sample —
        # traceability back to the source frame/observation, e.g. for error-analysis visualizations.

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
        offsets_grid_local_t = torch.from_numpy(self.offsets_grid_local).to(dev)
        self.flat_t, self.centers_t, self.pad_mask_t = flat_t, centers_t, pad_mask_t
        self.a1_t, self.a2_t, self.labels_t = a1_t, a2_t, labels_t
        self.offsets_grid_t = offsets_grid_t
        self.offsets_grid_local_t = offsets_grid_local_t
        self.device = dev
        return self

    def get_batch(self, idx):
        """Vectorized fetch — one gather for the whole batch, no Python per-sample loop.
        If to_device() was called, idx may be a numpy array or GPU long tensor; the gather
        and all outputs stay resident on GPU (no per-batch host->device transfer needed)."""
        if getattr(self, 'device', None) is not None:
            idx_t = idx if torch.is_tensor(idx) else torch.from_numpy(idx).to(self.device)
            window_idx = self.centers_t[idx_t].unsqueeze(1) + self.offsets_grid_local_t.unsqueeze(0)
            context = self.flat_t[window_idx].float()
            mask = self.pad_mask_t[idx_t]
            offs = self.offsets_grid_t.unsqueeze(0).expand(len(idx_t), len(self.offsets_grid_t))
            return context, offs, self.a1_t[idx_t], self.a2_t[idx_t], self.labels_t[idx_t], mask

        T = len(self.offsets_grid)
        window_idx = self.centers[idx][:, None] + self.offsets_grid_local[None, :]  # (B, T)
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


class FastFrameData:
    """Like FastBatchData, but one sample per annotated FRAME (not one per ordered
    pair) — for the per-frame classifier, which has no mouse-identity conditioning.

    Label is frame-level and multi-hot: [has_nt, has_nn], the OR over that frame's
    12 ordered-pair labels (a frame can contain both behaviors via different pairs
    at once, so this is multi-label, not 3-way single-label like the pairwise task).
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
        stride: int = 1,
    ):
        self.k = context_k
        self.stride = stride
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

        # frame-level multi-hot label: OR over that frame's positive ordered-pair rows
        # (pair_labels only stores positives, so a frame absent here is all-'none')
        frame_label = pair_labels.groupby('global_idx').agg(
            has_nt=('label', lambda s: bool((s == 1).any())),
            has_nn=('label', lambda s: bool((s == 2).any())),
        ).reset_index()

        all_global, all_obs_s, all_obs_e = [], [], []
        for oid, grp in ann_annotated.groupby('observation_id', sort=False):
            global_idxs = grp['index'].values.astype(np.int32)
            obs_s, obs_e = obs_boundary[oid]
            all_global.append(global_idxs)
            all_obs_s.append(np.full(len(global_idxs), obs_s, dtype=np.int32))
            all_obs_e.append(np.full(len(global_idxs), obs_e, dtype=np.int32))

        gi = np.concatenate(all_global)
        obs_s_arr = np.concatenate(all_obs_s)
        obs_e_arr = np.concatenate(all_obs_e)

        labels_df = pd.DataFrame({'global_idx': gi.astype(np.int32)}).merge(
            frame_label, on='global_idx', how='left'
        ).fillna(False)
        labels = labels_df[['has_nt', 'has_nn']].to_numpy(dtype=np.float32)  # (N, 2) multi-hot

        k = context_k
        reach = k * stride
        local = gi - obs_s_arr
        obs_len_arr = obs_e_arr - obs_s_arr

        print('  loading embeddings for vectorized batching...')
        obs_arrays = load_embeddings_fn(obs_boundary)  # {obs_s: (obs_len, ...) array}
        blocks = []
        cursor = 0
        pad_start_by_obs = {}
        for obs_s in sorted(obs_arrays.keys()):
            arr = obs_arrays[obs_s]
            pad_shape = (reach,) + arr.shape[1:]
            padded = np.concatenate(
                [np.zeros(pad_shape, dtype=arr.dtype), arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0
            )
            blocks.append(padded)
            pad_start_by_obs[obs_s] = cursor
            cursor += padded.shape[0]
        self.flat = (
            np.concatenate(blocks, axis=0) if blocks
            else np.zeros((0, emb_dim) if n_patches is None else (0, n_patches, emb_dim), dtype=np.float32)
        )
        del blocks

        center_adjust = np.array([pad_start_by_obs[s] for s in obs_s_arr], dtype=np.int64)
        self.centers = local + center_adjust  # position of this sample's offset=-reach frame
        self.offsets_grid_local = np.arange(0, 2 * reach + 1, stride, dtype=np.int64)
        offsets_grid = self.offsets_grid_local - reach
        self.offsets_grid = offsets_grid
        T = len(offsets_grid)
        abs_pos = local[:, None] + offsets_grid[None, :]
        self.pad_mask = (abs_pos < 0) | (abs_pos >= obs_len_arr[:, None])

        self.labels = labels  # (N, 2) float32 multi-hot: [has_nt, has_nn]
        self.gi = gi.astype(np.int64)

        n_pos = int((labels.sum(axis=1) > 0).sum())
        print(f'  {len(labels):,} frames | nt={int(labels[:, 0].sum()):,} nn={int(labels[:, 1].sum()):,} '
              f'(any-behavior={n_pos:,})')

    def __len__(self):
        return len(self.labels)

    def to_device(self, dev):
        """Same GPU-residency pattern as FastBatchData.to_device — see there for why."""
        flat_t = torch.from_numpy(self.flat).to(dev)
        centers_t = torch.from_numpy(self.centers).to(dev)
        pad_mask_t = torch.from_numpy(self.pad_mask).to(dev)
        labels_t = torch.from_numpy(self.labels).to(dev)
        offsets_grid_t = torch.from_numpy(self.offsets_grid).to(dev)
        offsets_grid_local_t = torch.from_numpy(self.offsets_grid_local).to(dev)
        self.flat_t, self.centers_t, self.pad_mask_t = flat_t, centers_t, pad_mask_t
        self.labels_t = labels_t
        self.offsets_grid_t = offsets_grid_t
        self.offsets_grid_local_t = offsets_grid_local_t
        self.device = dev
        return self

    def get_batch(self, idx):
        """Vectorized fetch — see FastBatchData.get_batch."""
        if getattr(self, 'device', None) is not None:
            idx_t = idx if torch.is_tensor(idx) else torch.from_numpy(idx).to(self.device)
            window_idx = self.centers_t[idx_t].unsqueeze(1) + self.offsets_grid_local_t.unsqueeze(0)
            context = self.flat_t[window_idx].float()
            mask = self.pad_mask_t[idx_t]
            offs = self.offsets_grid_t.unsqueeze(0).expand(len(idx_t), len(self.offsets_grid_t))
            return context, offs, self.labels_t[idx_t], mask

        T = len(self.offsets_grid)
        window_idx = self.centers[idx][:, None] + self.offsets_grid_local[None, :]  # (B, T)
        context = self.flat[window_idx]
        mask = self.pad_mask[idx]
        offs = np.broadcast_to(self.offsets_grid, (len(idx), T))
        return (
            torch.from_numpy(np.ascontiguousarray(context)),
            torch.from_numpy(np.ascontiguousarray(offs)).long(),
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
