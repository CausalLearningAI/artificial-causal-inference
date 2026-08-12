"""
One-off: compute full-train-set macro AP for an already-trained patchgrid256
model (never computed during training — only train_loss was tracked, and AP
was only ever evaluated on val). Reloads the saved checkpoint, rebuilds the
same train split/bound, re-encodes the needed frames, and scores them.

Usage:
    python scripts/mice_behavior/eval_train_ap.py --encoder dinov2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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

MODEL_IDS = {'dinov2': 'facebook/dinov2-base', 'dinov3': 'facebook/dinov3-vitb16-pretrain-lvd1689m'}
N_PATCHES = {'dinov2': 256, 'dinov3': 196}
EMB_DIM = 768

p = argparse.ArgumentParser()
p.add_argument('--encoder', required=True, choices=sorted(MODEL_IDS))
p.add_argument('--max-train-frames', type=int, default=200_000)
p.add_argument('--batch-size', type=int, default=256)
args = p.parse_args()
MODEL_ID = MODEL_IDS[args.encoder]
n_patches_full = N_PATCHES[args.encoder]

OUT_DIR = gsf.FRAME_DIR / f'patchgrid256_{args.encoder}'
cfg_all = json.load(open(OUT_DIR / 'config.json'))
best_cfg = cfg_all['cfg']
print(f'Evaluating {OUT_DIR}/best_model.pt (cfg={best_cfg}) on its own TRAIN split...', flush=True)

pair_labels_path = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
annotations_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
obs_to_pool = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
pools = sorted({obs_to_pool[o] for o in all_obs})
val_pool_set = get_val_pools(pools, seed=gsf.SEED)
train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
assert sorted(val_pool_set) == cfg_all['val_pools'], 'split mismatch vs. saved config!'

train_meta = FrameBatchData(
    str(annotations_csv), str(pair_labels_path), train_obs, best_cfg['context_k'], 1,
    dummy_loader(1, 1), n_patches=1, stride=best_cfg['stride'], max_frames=args.max_train_frames, seed=gsf.SEED,
)
del train_meta.flat
print(f'{len(train_meta):,} train samples (bounded by max_train_frames={args.max_train_frames})', flush=True)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
encoder = AutoModel.from_pretrained(MODEL_ID).to(dev)
encoder.eval()
encoder.requires_grad_(False)
n_register_tokens = getattr(encoder.config, 'num_register_tokens', 0)
n_prefix_tokens = 1 + n_register_tokens

hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(gsf.DATASET_DIR), frame_type='full')

abs_idx_all = train_meta.gi[:, None] + train_meta.offsets_grid[None, :]
need = np.unique(abs_idx_all[~train_meta.pad_mask])
print(f'Encoding {len(need):,} unique train frames...', flush=True)
t0 = time.time()
loader = DataLoader(_ImageDataset(hf_dataset, need, processor), batch_size=256, num_workers=16,
                     pin_memory=(dev.type == 'cuda'), shuffle=False, prefetch_factor=4, persistent_workers=True)
cache = torch.empty((len(need), n_patches_full, EMB_DIM), dtype=torch.float16)
idx_of_global = {int(g): i for i, g in enumerate(need)}
cursor = 0
with torch.inference_mode():
    for pixel_values in loader:
        pixel_values = pixel_values.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
            out = encoder(pixel_values=pixel_values)
        tokens = out.last_hidden_state[:, n_prefix_tokens:].half().cpu()
        cache[cursor:cursor + tokens.shape[0]] = tokens
        cursor += tokens.shape[0]
print(f'Encoding done in {(time.time()-t0)/60:.1f} min', flush=True)
del encoder, processor, loader
if dev.type == 'cuda':
    torch.cuda.empty_cache()

model = MouseFrameClassifier(
    emb_dim=EMB_DIM, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'],
    use_patch_grid=True, dropout=best_cfg['dropout'],
).to(dev)
model.load_state_dict(torch.load(OUT_DIR / 'best_model.pt', map_location=dev, weights_only=True))
model.eval()

offsets = train_meta.offsets_grid
all_probs, all_labels = [], []
with torch.no_grad():
    for b0 in range(0, len(train_meta), args.batch_size):
        sample_idx = np.arange(b0, min(b0 + args.batch_size, len(train_meta)))
        gi = train_meta.gi[sample_idx]
        abs_idx = gi[:, None] + offsets[None, :]
        mask = train_meta.pad_mask[sample_idx]
        B, T = abs_idx.shape
        valid = ~mask
        flat_idx = abs_idx[valid]
        positions = np.array([idx_of_global[int(g)] for g in flat_idx], dtype=np.int64)
        gathered = cache[positions].to(dev, non_blocking=True)
        ctx = torch.zeros((B, T, n_patches_full, EMB_DIM), dtype=torch.float16, device=dev)
        ctx[torch.from_numpy(valid)] = gathered
        offsets_t = torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()).to(dev)
        mask_t = torch.from_numpy(mask).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
            logits = model(ctx.float(), offsets=offsets_t, key_padding_mask=mask_t)
        all_probs.append(torch.sigmoid(logits).float().cpu())
        all_labels.append(torch.from_numpy(train_meta.labels[sample_idx]))

probs = torch.cat(all_probs).numpy()
labels = torch.cat(all_labels).numpy()
per_label = {name: average_precision_score(labels[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
macro_ap = float(np.mean(list(per_label.values())))
print(f'\nTRAIN-set macro AP: {macro_ap:.4f}  {per_label}', flush=True)
print(f'(for comparison, VAL-set macro AP was {cfg_all["best_ap"]:.4f}  {cfg_all["best_per_label"]})', flush=True)
