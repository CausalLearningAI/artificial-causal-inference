"""
One-off: generate the missing report.png/roc_pr_data.npz for patchgrid256_dinov2
(train_patchgrid_online.py never called generate_frame_report). Recomputes full-val
probs/labels from the saved best_model.pt and re-encodes the (small) val set once.

Usage:
    python scripts/mice_behavior/backfill_online_report.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.report import generate_frame_report
from src.dataset.get_dataset import load_dataset
from transformers import AutoImageProcessor, AutoModel
from train_patchgrid_online import dummy_loader, _ImageDataset, N_PATCHES_FULL, EMB_DIM, MODEL_ID

OUT_DIR = gsf.FRAME_DIR / 'patchgrid256_dinov2'
cfg_all = json.load(open(OUT_DIR / 'config.json'))
best_cfg = cfg_all['cfg']
history = json.load(open(OUT_DIR / 'history.json'))
history['eval_epoch'] = history['epoch']  # evaluated every epoch in this script

pair_labels_path = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
annotations_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
obs_to_pool = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
pools = sorted({obs_to_pool[o] for o in all_obs})
rng_split = random.Random(gsf.SEED)
shuffled = pools[:]
rng_split.shuffle(shuffled)
n_val = max(1, int(len(shuffled) * 0.2))
val_pool_set = set(shuffled[:n_val])
val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]

val_meta = FrameBatchData(
    str(annotations_csv), str(pair_labels_path), val_obs, best_cfg['context_k'], 1,
    dummy_loader(1, 1), n_patches=1, stride=best_cfg['stride'],
)
del val_meta.flat

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
encoder = AutoModel.from_pretrained(MODEL_ID).to(dev)
encoder.eval()
encoder.requires_grad_(False)
hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(gsf.DATASET_DIR), frame_type='full')

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

model = MouseFrameClassifier(
    emb_dim=EMB_DIM, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'],
    use_patch_grid=True, dropout=best_cfg['dropout'],
).to(dev)
model.load_state_dict(torch.load(OUT_DIR / 'best_model.pt', map_location=dev, weights_only=True))
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
per_label = {name: average_precision_score(labels[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
print('Recomputed full-val AP (sanity check vs config.json):', per_label)

generate_frame_report(probs, labels, history,
                       'Patch-grid DINOv2, all 256 raw patch tokens (online, no pooling)', best_cfg, OUT_DIR)
print(f'Saved {OUT_DIR}/report.png (+ roc_pr_data.npz)')
