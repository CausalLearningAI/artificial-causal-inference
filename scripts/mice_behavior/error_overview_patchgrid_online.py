"""
Visual error overview for patchgrid256_dinov2 (all 256 raw DINOv2 patch tokens,
online-computed) — same k=10-examples-per-confusion-bucket format as
error_overview_frame.py (the CLS variant's own script), adapted to re-encode the
needed val frames online instead of reading cached embeddings, since this variant
never caches its full-resolution representation to disk.

Usage:
    python scripts/mice_behavior/error_overview_patchgrid_online.py
"""
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.dataset.get_dataset import load_dataset
from train_patchgrid_online import dummy_loader, _ImageDataset, N_PATCHES_FULL, EMB_DIM, MODEL_ID

DATA_DIR = gsf.DATA_DIR
DATASET_DIR = gsf.DATASET_DIR
SEED = gsf.SEED
K = 10
BUCKETS = ['TP', 'TN', 'FP', 'FN']
BEHAVIORS = [(0, 'nt', 'Nose-Tail Sniffing'), (1, 'nn', 'Nose-Nose Sniffing')]


def fmt_pct(pct):
    return f'{pct:.0f}%' if pct >= 1 else f'{pct:.2f}%'


def main():
    out_dir = gsf.FRAME_DIR / 'patchgrid256_dinov2'
    cfg = json.load(open(out_dir / 'config.json'))['cfg']

    pair_labels_path = gsf.build_pair_labels(DATA_DIR, DATASET_DIR, overwrite=False)
    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    obs_to_pool = gsf.load_obs_to_pool_map(DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})
    rng = random.Random(SEED)
    shuffled = pools[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.2))
    val_pool_set = set(shuffled[:n_val])
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]

    val_meta = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, cfg['context_k'], 1,
        dummy_loader(1, 1), n_patches=1, stride=cfg['stride'],
    )
    del val_meta.flat

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    encoder = AutoModel.from_pretrained(MODEL_ID).to(dev)
    encoder.eval()
    encoder.requires_grad_(False)
    hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(DATASET_DIR), frame_type='full')

    need = np.unique(val_meta.gi[:, None] + val_meta.offsets_grid[None, :])
    print(f'Encoding {len(need):,} unique val frames...', flush=True)
    loader = DataLoader(_ImageDataset(hf_dataset, need, processor), batch_size=256, num_workers=8,
                         pin_memory=(dev.type == 'cuda'), shuffle=False)
    cache = torch.empty((len(need), N_PATCHES_FULL, EMB_DIM), dtype=torch.float16)
    idx_of_global = {int(g): i for i, g in enumerate(need)}
    cursor = 0
    with torch.inference_mode():
        for pixel_values in loader:
            pixel_values = pixel_values.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
                out = encoder(pixel_values=pixel_values)
            tokens = out.last_hidden_state[:, 1:].half().cpu()
            cache[cursor:cursor + tokens.shape[0]] = tokens
            cursor += tokens.shape[0]
    print('Encoding done.', flush=True)
    del encoder, processor, loader
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    model = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'],
        use_patch_grid=True, dropout=cfg['dropout'],
    ).to(dev)
    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=dev, weights_only=True))
    model.eval()

    all_probs, all_labels = [], []
    BATCH = 256
    with torch.no_grad():
        for b0 in range(0, len(val_meta), BATCH):
            sample_idx = np.arange(b0, min(b0 + BATCH, len(val_meta)))
            gi = val_meta.gi[sample_idx]
            offsets = val_meta.offsets_grid
            abs_idx = gi[:, None] + offsets[None, :]
            mask = val_meta.pad_mask[sample_idx]
            B, T = abs_idx.shape
            valid = ~mask
            flat_idx = abs_idx[valid]
            positions = np.array([idx_of_global[int(g)] for g in flat_idx], dtype=np.int64)
            gathered = cache[positions].to(dev, non_blocking=True)
            ctx = torch.zeros((B, T, N_PATCHES_FULL, EMB_DIM), dtype=torch.float16, device=dev)
            ctx[torch.from_numpy(valid)] = gathered
            offsets_t = torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()).to(dev)
            mask_t = torch.from_numpy(mask).to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
                logits = model(ctx.float(), offsets=offsets_t, key_padding_mask=mask_t)
            all_probs.append(torch.sigmoid(logits).float().cpu())
            all_labels.append(torch.from_numpy(val_meta.labels[sample_idx]))

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    pred = probs > 0.5
    gi = val_meta.gi
    n_total = len(labels)

    frame_paths = pd.read_csv(annotations_csv, usecols=['frame_path'])['frame_path'].values

    row_heights, row_plan = [], []
    for bi, (behavior_c, behavior_name, behavior_title) in enumerate(BEHAVIORS):
        row_heights.append(0.45); row_plan.append(('title', behavior_c, behavior_title))
        for bucket in BUCKETS:
            row_heights.append(1.0); row_plan.append(('bucket', behavior_c, bucket))
        if bi < len(BEHAVIORS) - 1:
            row_heights.append(0.35); row_plan.append(('spacer', None, None))

    fig = plt.figure(figsize=(2 * K, 2 * sum(row_heights)))
    gs = fig.add_gridspec(len(row_heights), K, height_ratios=row_heights, hspace=0.15, wspace=0.05)
    pick_rng = np.random.default_rng(SEED)

    for r, (kind, behavior_c, payload) in enumerate(row_plan):
        if kind == 'spacer':
            continue
        if kind == 'title':
            ax = fig.add_subplot(gs[r, :])
            ax.axis('off')
            ax.text(0.5, 0.5, f'{payload} (validation set)', ha='center', va='center', fontsize=13, fontweight='bold')
            continue

        bucket = payload
        is_true = labels[:, behavior_c].astype(bool)
        is_pred = pred[:, behavior_c]
        mask = {
            'TP': is_true & is_pred, 'TN': ~is_true & ~is_pred,
            'FP': ~is_true & is_pred, 'FN': is_true & ~is_pred,
        }[bucket]
        idx = np.where(mask)[0]
        pct = 100 * len(idx) / n_total
        row_label = f'{bucket}\n({fmt_pct(pct)})'
        chosen = pick_rng.choice(idx, size=min(K, len(idx)), replace=False) if len(idx) else np.array([], dtype=int)

        for col in range(K):
            ax = fig.add_subplot(gs[r, col])
            ax.axis('off')
            if col < len(chosen):
                i = chosen[col]
                img_path = DATASET_DIR / frame_paths[gi[i]]
                try:
                    ax.imshow(Image.open(img_path))
                except Exception:
                    ax.text(0.5, 0.5, '(image not found)', ha='center', va='center', fontsize=6, transform=ax.transAxes)
                ax.set_title(f'p={probs[i, behavior_c]:.2f}', fontsize=7)
            if col == 0:
                ax.text(-0.15, 0.5, row_label, transform=ax.transAxes, fontsize=9, fontweight='bold',
                        ha='right', va='center', rotation=0)

    fig.suptitle(f'patchgrid256_dinov2: error overview (p>0.5 decision rule, k={K} random examples per bucket)', fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.02)
    out_path = out_dir / 'error_overview.png'
    fig.savefig(out_path, dpi=120)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
