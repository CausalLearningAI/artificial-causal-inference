"""
One-off: finalize patchgrid256_dinov2 using the best config already found by
search_patchgrid_online.py's 14 completed trials (trial 5 — the search job hit its
8h time limit one trial short of finishing and never reached its own promotion
step). Retrains this one known-best config on the standard (non-fold) train/val
split and saves model+config.json, matching what the search would have produced.

Usage:
    python scripts/mice_behavior/finalize_patchgrid_dinov2.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_val_pools
from src.dataset.get_dataset import load_dataset
from train_patchgrid_online import dummy_loader, _ImageDataset

MODEL_ID = 'facebook/dinov2-base'
N_PATCHES_FULL = 256
EMB_DIM = 768
MAX_TRAIN_FRAMES = 200_000
N_EPOCHS = 20
PATIENCE = 7
BATCH_SIZE = 256

# Best trial from the 14 completed before the search hit its time limit
# (logs/search_patchgrid256_62600323.out, tag patchgrid256_dinov2_5, AP=0.2537).
BEST_CFG = dict(n_heads=4, context_k=2, stride=1, hidden_dim=128, neg_ratio=15,
                 dropout=0.1, weight_decay=0.0001, lr=0.001)

OUT_DIR = gsf.FRAME_DIR / 'patchgrid256_dinov2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

pair_labels_path = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
annotations_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
obs_to_pool = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
pools = sorted({obs_to_pool[o] for o in all_obs})
val_pool_set = get_val_pools(pools, seed=gsf.SEED)
train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
print(f'Split: {len(train_obs)} train obs / {len(val_obs)} val obs (val_pools={sorted(val_pool_set)})', flush=True)

train_meta = FrameBatchData(
    str(annotations_csv), str(pair_labels_path), train_obs, BEST_CFG['context_k'], 1,
    dummy_loader(1, 1), n_patches=1, stride=BEST_CFG['stride'], max_frames=MAX_TRAIN_FRAMES, seed=gsf.SEED,
)
val_meta = FrameBatchData(
    str(annotations_csv), str(pair_labels_path), val_obs, BEST_CFG['context_k'], 1,
    dummy_loader(1, 1), n_patches=1, stride=BEST_CFG['stride'],
)
del train_meta.flat, val_meta.flat

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading {MODEL_ID} on {dev}...', flush=True)
processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
encoder = AutoModel.from_pretrained(MODEL_ID).to(dev)
encoder.eval()
encoder.requires_grad_(False)

print('Loading full-frame HF dataset (raw JPEGs)...', flush=True)
hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(gsf.DATASET_DIR), frame_type='full')


def _needed_raw_indices(meta):
    abs_idx = meta.gi[:, None] + meta.offsets_grid[None, :]
    return np.unique(abs_idx[~meta.pad_mask])


all_needed = np.unique(np.concatenate([_needed_raw_indices(train_meta), _needed_raw_indices(val_meta)]))
print(f'Encoding {len(all_needed):,} unique frames...', flush=True)
t0 = time.time()
loader = DataLoader(
    _ImageDataset(hf_dataset, all_needed, processor), batch_size=256, num_workers=16,
    pin_memory=(dev.type == 'cuda'), shuffle=False, prefetch_factor=4, persistent_workers=True,
)
cache = torch.empty((len(all_needed), N_PATCHES_FULL, EMB_DIM), dtype=torch.float16)
idx_of_global = {int(g): i for i, g in enumerate(all_needed)}
cursor = 0
with torch.inference_mode():
    for pixel_values in loader:
        pixel_values = pixel_values.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
            out = encoder(pixel_values=pixel_values)
        tokens = out.last_hidden_state[:, 1:].half().cpu()
        cache[cursor:cursor + tokens.shape[0]] = tokens
        cursor += tokens.shape[0]
print(f'Encoding done in {(time.time()-t0)/60:.1f} min', flush=True)
del encoder, processor, loader
if dev.type == 'cuda':
    torch.cuda.empty_cache()

offsets = train_meta.offsets_grid


def build_batch_tensor(meta, sample_idx):
    gi = meta.gi[sample_idx]
    abs_idx = gi[:, None] + offsets[None, :]
    mask = meta.pad_mask[sample_idx]
    B, T = abs_idx.shape
    valid = ~mask
    flat_idx = abs_idx[valid]
    positions = np.array([idx_of_global[int(g)] for g in flat_idx], dtype=np.int64)
    gathered = cache[positions].to(dev, non_blocking=True)
    ctx = torch.zeros((B, T, N_PATCHES_FULL, EMB_DIM), dtype=torch.float16, device=dev)
    ctx[torch.from_numpy(valid)] = gathered
    offsets_t = torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()).to(dev)
    labels_t = torch.from_numpy(meta.labels[sample_idx]).to(dev)
    mask_t = torch.from_numpy(mask).to(dev)
    return ctx, offsets_t, labels_t, mask_t


labels_all = train_meta.labels
pos_idx = np.where(labels_all.sum(axis=1) > 0)[0]
neg_idx = np.where(labels_all.sum(axis=1) == 0)[0]
n_pos = max(len(pos_idx), 1)
n_neg_draw = min(len(neg_idx), BEST_CFG['neg_ratio'] * n_pos)
rng = np.random.default_rng(gsf.SEED)
neg_sample = rng.choice(neg_idx, size=n_neg_draw, replace=False)
epoch_idx_fixed = np.concatenate([pos_idx, neg_sample])

pos_counts = labels_all[pos_idx].sum(axis=0).clip(min=1)
pos_weight = torch.tensor(max(n_neg_draw, 1) / pos_counts, dtype=torch.float32).to(dev)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

lr = min(BEST_CFG['lr'], 3e-4)
model = MouseFrameClassifier(
    emb_dim=EMB_DIM, n_heads=BEST_CFG['n_heads'], hidden_dim=BEST_CFG['hidden_dim'],
    use_patch_grid=True, dropout=BEST_CFG['dropout'],
).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=BEST_CFG['weight_decay'])
amp_enabled = dev.type == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

shuffle_rng = np.random.default_rng(gsf.SEED)
best_ap, epochs_since_best, best_state = -1.0, 0, None
for epoch in range(1, N_EPOCHS + 1):
    model.train()
    epoch_idx = epoch_idx_fixed.copy()
    shuffle_rng.shuffle(epoch_idx)
    for b0 in range(0, len(epoch_idx), BATCH_SIZE):
        batch_idx = epoch_idx[b0:b0 + BATCH_SIZE]
        ctx, offs, lbl, mask = build_batch_tensor(train_meta, batch_idx)
        optimizer.zero_grad()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
            loss = criterion(logits, lbl)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for b0 in range(0, len(val_meta), BATCH_SIZE):
            batch_idx = np.arange(b0, min(b0 + BATCH_SIZE, len(val_meta)))
            ctx, offs, lbl, mask = build_batch_tensor(val_meta, batch_idx)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
            all_probs.append(torch.sigmoid(logits).float().cpu())
            all_labels.append(lbl.cpu())
    probs = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    macro_ap = float(np.mean([average_precision_score(labels_np[:, i], probs[:, i]) for i in range(2)]))
    print(f'epoch {epoch}/{N_EPOCHS}  macro_ap={macro_ap:.4f}', flush=True)

    if macro_ap > best_ap:
        best_ap, epochs_since_best = macro_ap, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        epochs_since_best += 1
        if epochs_since_best >= PATIENCE:
            print(f'early stopping: no improvement for {PATIENCE} epochs', flush=True)
            break

model.load_state_dict(best_state)
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for b0 in range(0, len(val_meta), BATCH_SIZE):
        batch_idx = np.arange(b0, min(b0 + BATCH_SIZE, len(val_meta)))
        ctx, offs, lbl, mask = build_batch_tensor(val_meta, batch_idx)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
            logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
        all_probs.append(torch.sigmoid(logits).float().cpu())
        all_labels.append(lbl.cpu())
probs = torch.cat(all_probs).numpy()
labels_np = torch.cat(all_labels).numpy()
per_label = {name: average_precision_score(labels_np[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
final_ap = float(np.mean(list(per_label.values())))
print(f'FINAL full-val macro AP: {final_ap:.4f}  {per_label}', flush=True)

torch.save(best_state, OUT_DIR / 'best_model.pt')
with open(OUT_DIR / 'config.json', 'w') as f:
    json.dump({'cfg': BEST_CFG, 'val_pools': sorted(val_pool_set), 'n_patches': N_PATCHES_FULL,
               'max_train_frames': MAX_TRAIN_FRAMES, 'best_ap': final_ap, 'best_per_label': per_label,
               'encoder': 'dinov2', 'note': 'finalized from 14/15 completed search trials (job hit 8h time limit)'}, f, indent=2)
print(f'Saved {OUT_DIR}/{{best_model.pt,config.json}}', flush=True)
