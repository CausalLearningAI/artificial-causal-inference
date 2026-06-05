"""
Dataset for mouse pairwise behavior classification.

Each sample is (context_seq, mouse1_idx, mouse2_idx) → label in {0=none, 1=nt, 2=nn}.
context_seq is the raw embedding sequence in the window [t-k, t+k] (shape T×d, T≤2k+1),
clamped to observation boundaries. Use collate_fn to batch variable-length sequences.
Mouse indices (0-3) index into 4 globally shared learnable query vectors.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MousePairDataset(Dataset):
    def __init__(
        self,
        annotations_csv: str,
        pair_labels_parquet: str,
        embeddings_path: str,
        obs_ids=None,
        context_k: int = 2,
        emb_dim: int = 768,
        preload: bool = True,
    ):
        self.k = context_k

        # Row index in annotations.csv == row index in embeddings.npy (both built in same order)
        print('  loading annotations index...')
        ann = pd.read_csv(annotations_csv, usecols=['observation_id', 'frame_idx'])

        if obs_ids is not None:
            obs_set = set(obs_ids)
            ann = ann[ann['observation_id'].isin(obs_set)]

        # obs_boundary[obs_id] = (global_start, global_end)  — [start, end) in embedding array
        ann_reset = ann.reset_index()  # preserves original row number as 'index' (= embedding row)
        obs_boundary = {}
        for oid, grp in ann_reset.groupby('observation_id', sort=False):
            idx = grp['index'].values
            obs_boundary[oid] = (int(idx[0]), int(idx[-1]) + 1)
        self.obs_boundary = obs_boundary

        # Load positive pair labels and map to global embedding indices
        print('  loading pair labels...')
        pair_labels = pd.read_parquet(pair_labels_parquet)
        if obs_ids is not None:
            pair_labels = pair_labels[pair_labels['observation_id'].isin(obs_set)]

        frame_to_global = ann_reset[['observation_id', 'frame_idx', 'index']].rename(
            columns={'index': 'global_idx'}
        )
        pair_labels = pair_labels.merge(
            frame_to_global, on=['observation_id', 'frame_idx'], how='inner'
        ).astype({'global_idx': np.int32})

        # Build samples: annotated frames × 12 ordered pairs
        print('  building sample index...')
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
        obs_s = np.concatenate(all_obs_s)
        obs_e = np.concatenate(all_obs_e)

        # Fill in positive labels via merge
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

        # Store as int32 array: [global_idx, a1, a2, label, obs_start, obs_end]
        self.samples = np.column_stack(
            [gi, a1, a2, labels.astype(np.int32), obs_s, obs_e]
        ).astype(np.int32)

        # Load embeddings: preload annotated obs into RAM (avoids NFS random I/O during training),
        # or fall back to mmap for inference / low-memory cases.
        emb_path = Path(embeddings_path)
        n_total = emb_path.stat().st_size // (4 * emb_dim)
        raw_mmap = np.memmap(emb_path, dtype='float32', mode='r', shape=(n_total, emb_dim))

        if preload:
            print('  preloading embeddings into RAM (sequential NFS read)...')
            # Load each annotated obs block contiguously — O(n_annotated_frames) sequential reads
            self._obs_arrays = {}  # obs_start → dense float32 array
            for oid in annotated_obs:
                obs_s, obs_e = obs_boundary[oid]
                self._obs_arrays[obs_s] = np.array(raw_mmap[obs_s:obs_e])  # copy into RAM
            total_mb = sum(a.nbytes for a in self._obs_arrays.values()) / 1e6
            print(f'  preloaded {total_mb:.0f} MB')
            self._preloaded = True
        else:
            self.embeddings = raw_mmap
            self._preloaded = False

        # Per-sample weights for balanced class sampling
        unique, counts = np.unique(labels, return_counts=True)
        n_classes = 3
        w = {int(c): len(labels) / (n_classes * cnt) for c, cnt in zip(unique, counts)}
        self.sample_weights = np.array([w.get(int(l), 1.0) for l in labels], dtype=np.float64)

        print(
            f'  {len(self.samples):,} samples | '
            + ' '.join(f'label{c}={cnt:,}' for c, cnt in zip(unique, counts))
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gi, a1, a2, label, obs_s, obs_e = self.samples[idx]
        k = self.k
        if self._preloaded:
            local = int(gi) - int(obs_s)
            obs_len = int(obs_e) - int(obs_s)
            lo = max(0, local - k)
            hi = min(obs_len - 1, local + k)
            context = self._obs_arrays[int(obs_s)][lo : hi + 1]  # (T, d)
        else:
            lo = max(int(obs_s), int(gi) - k)
            hi = min(int(obs_e) - 1, int(gi) + k)
            context = self.embeddings[lo : hi + 1]  # (T, d)
        return (
            torch.from_numpy(context.copy()),  # (T, d)
            torch.tensor(int(a1), dtype=torch.long),
            torch.tensor(int(a2), dtype=torch.long),
            torch.tensor(int(label), dtype=torch.long),
        )


def collate_fn(batch):
    """Pad variable-length sequences and build key_padding_mask for cross-attention."""
    seqs, a1s, a2s, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs])
    # pad_sequence: list of (T_i, d) → (B, T_max, d)
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    T_max = padded.size(1)
    # True = padding position (ignored by MultiheadAttention)
    key_padding_mask = torch.arange(T_max).unsqueeze(0) >= lengths.unsqueeze(1)
    return (
        padded,
        torch.stack(a1s),
        torch.stack(a2s),
        torch.stack(labels),
        key_padding_mask,
    )
