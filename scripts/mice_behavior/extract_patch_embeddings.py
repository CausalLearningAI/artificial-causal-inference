"""
Extract coarse spatial patch-grid embeddings (DINOv2) for the annotated mice v1
observations only — not the full dataset. Used to test whether replacing the
CLS-token summary with a small pooled patch grid + attention improves the
pairwise behavior classifier, before committing to full-dataset extraction.

Output: dataset/mice/v1/embeddings/full/dinov2/patch_grid{G}/
    embeddings.npy   — fp16, shape (n_annotated_frames, G*G, emb_dim)
    global_idx.npy   — int32, shape (n_annotated_frames,); row i was extracted from
                        frame-table row global_idx[i] AT EXTRACTION TIME (this goes
                        stale if annotations.csv is rebuilt -- do not index by it).
    row_keys.parquet — (observation_id, frame_idx) of every row, in row order. This is
                        what loaders use (src/mice_behavior/emb_index.py).

Usage:
    python scripts/mice_behavior/extract_patch_embeddings.py --grid-size 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Installed Pillow (9.2.0) predates PIL.Image.ExifTags.Base, which
# `datasets` 4.5.0's image decoder unconditionally references. Shim the
# standard EXIF orientation tag (274) so dataset['image'] decoding works
# without touching the shared conda environment.
import PIL.Image
if not hasattr(PIL.Image, 'ExifTags') or not hasattr(PIL.Image.ExifTags, 'Base'):
    class _ExifTagsBaseShim:
        Orientation = 274
    class _ExifTagsShim:
        Base = _ExifTagsBaseShim
    PIL.Image.ExifTags = _ExifTagsShim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataset.get_dataset import load_dataset
from src.mice_behavior.emb_index import hf_column, write_row_keys

MODEL_IDS = {
    'dinov2': 'facebook/dinov2-base',
    'dinov3': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
}
EMB_DIM = 768


def patch_grid_forward(model, pixel_values, n_prefix_tokens: int, G: int, autocast_ctx):
    """(B, G*G, D) float32: last_hidden_state minus CLS (+ register) tokens, adaptive-avg-pooled
    to a G x G grid. Shared with complete_emb_caches.py so appended rows use the same maths."""
    with torch.inference_mode(), autocast_ctx:
        outputs = model(pixel_values=pixel_values)
    patch_tokens = outputs.last_hidden_state[:, n_prefix_tokens:].float()  # (B, N_patches, emb_dim), drop CLS (+ registers)
    B, N, D = patch_tokens.shape
    side = int(round(N ** 0.5))
    assert side * side == N, f'patch count {N} is not a perfect square, cannot reshape to a grid'
    grid = patch_tokens.transpose(1, 2).reshape(B, D, side, side)  # (B, D, side, side)
    pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (G, G))  # (B, D, G, G)
    return pooled.reshape(B, D, G * G).transpose(1, 2)  # (B, G*G, D)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--dataset-dir', default='./dataset')
    p.add_argument('--encoder', default='dinov2', choices=sorted(MODEL_IDS), help='which backbone to extract patch tokens from')
    p.add_argument('--grid-size', type=int, default=4, help='pooled grid is grid-size x grid-size')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    MODEL_ID = MODEL_IDS[args.encoder]

    dataset_dir = Path(args.dataset_dir)
    out_dir = dataset_dir / 'mice' / 'v1' / 'embeddings' / 'full' / args.encoder / f'patch_grid{args.grid_size}'
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_out = out_dir / 'embeddings.npy'
    idx_out = out_dir / 'global_idx.npy'

    if emb_out.exists() and idx_out.exists() and not args.overwrite:
        print(f'[SKIP] {emb_out} already exists — use --overwrite to recompute')
        return

    # --- Determine which global frame indices belong to annotated observations ---
    print('Loading pair_labels + annotations index...')
    pair_labels = pd.read_parquet(dataset_dir / 'mice' / 'v1' / 'pair_labels.parquet')
    annotated_obs = set(pair_labels['observation_id'].unique())

    ann = pd.read_csv(dataset_dir / 'mice' / 'v1' / 'annotations.csv', usecols=['observation_id', 'frame_idx'])
    ann_reset = ann.reset_index()  # 'index' column == annotations.csv row (== HF frame-dataset row, checked below)
    sub = ann_reset[ann_reset['observation_id'].isin(annotated_obs)]
    global_idx = sub['index'].values.astype(np.int32)
    n_frames = len(global_idx)
    print(f'  {len(annotated_obs)} annotated observations -> {n_frames:,} frames to extract '
          f'(of {len(ann_reset):,} total v1 frames)')

    # --- Load the full HF frame dataset (avoid .select(): this repo's custom
    # DatasetInfo (with a 'metadata' field) breaks datasets' internal
    # info.copy() call inside .select()/_select_contiguous()) ---
    print('Loading full-frame dataset (indexing directly, no .select())...')
    hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(dataset_dir), frame_type='full')
    # The frames are read from hf_dataset[global_idx]; the keys written to row_keys.parquet come
    # from annotations.csv. Refuse to run if the two tables disagree on those rows.
    if len(hf_dataset) != len(ann):
        raise RuntimeError(f'HF frame dataset has {len(hf_dataset):,} rows but annotations.csv has {len(ann):,}')
    hf_oid = hf_column(hf_dataset, 'observation_id')[global_idx]
    hf_fidx = hf_column(hf_dataset, 'frame_idx').astype(np.int64)[global_idx]
    if not ((hf_oid == sub['observation_id'].values).all() and (hf_fidx == sub['frame_idx'].values).all()):
        raise RuntimeError('HF frame dataset and annotations.csv disagree on (observation_id, frame_idx) '
                           'for the selected rows -- rebuild one of them before extracting')

    # --- Model ---
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Loading {MODEL_ID} on {device}...')
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    model.requires_grad_(False)

    # DINOv3 (and DINOv2's *_reg variants) prepend `num_register_tokens` storage tokens
    # between CLS and the patch tokens in last_hidden_state — skip those too, or the
    # reshape-to-grid below silently misaligns (or fails the perfect-square assert).
    n_register_tokens = getattr(model.config, 'num_register_tokens', 0)
    n_prefix_tokens = 1 + n_register_tokens
    if n_register_tokens:
        print(f'  {MODEL_ID} has {n_register_tokens} register tokens — skipping CLS + registers ({n_prefix_tokens} tokens) before the patch grid')

    G = args.grid_size

    class _PreprocessedDataset(torch.utils.data.Dataset):
        """Indexes hf_dataset via global_idx directly — never calls .select()."""

        def __init__(self, ds, indices, proc):
            self.ds, self.indices, self.proc = ds, indices, proc

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            image = self.ds[int(self.indices[idx])]['image']
            processed = self.proc(images=image, return_tensors='pt')
            return {k: v.squeeze(0) for k, v in processed.items()}

    loader = torch.utils.data.DataLoader(
        _PreprocessedDataset(hf_dataset, global_idx, processor),
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), shuffle=False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    tmp_emb = Path(str(emb_out) + '.tmp')
    emb_mmap = np.memmap(tmp_emb, dtype='float16', mode='w+', shape=(n_frames, G * G, EMB_DIM))

    current = 0
    use_fp16 = device.type == 'cuda'
    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16)
    with tqdm(total=n_frames, desc=f'Extracting patch_grid{G}', unit='frame') as pbar:
        for batch in loader:
            pooled = patch_grid_forward(model, batch['pixel_values'].to(device), n_prefix_tokens, G, autocast_ctx)

            batch_len = pooled.shape[0]
            emb_mmap[current:current + batch_len] = pooled.half().cpu().numpy()
            current += batch_len
            pbar.update(batch_len)

    emb_mmap.flush()
    tmp_emb.rename(emb_out)
    np.save(idx_out, global_idx)
    keys_out = write_row_keys(out_dir, sub['observation_id'].values, sub['frame_idx'].values)
    print(f'\n✓ Saved {emb_out} ({emb_out.stat().st_size / 1e9:.1f} GB), {idx_out} and {keys_out}')


if __name__ == '__main__':
    main()
